// llama-sortformer — ggml/CUDA port of NVIDIA Streaming Sortformer (speaker diarization).
//
// M4 progress: scaffold + GGUF loader + CUDA backend + log-mel (validated) + the Conformer
// subsampling stem (dw_striding), validated against the NeMo reference (reference_offline/).
// Conformer blocks, transformer head and AOSC streaming follow. Not an LLM (links ggml only).

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "gguf.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <complex>
#include <fstream>
#include <stdexcept>

// ----------------------------------------------------------------- tiny .npy reader (f32)
struct NpyF32 { std::vector<int64_t> shape; std::vector<float> data;
    int64_t numel() const { int64_t n = 1; for (auto s : shape) n *= s; return n; } };

static NpyF32 npy_load_f32(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open npy: " + path);
    char magic[6]; f.read(magic, 6);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) throw std::runtime_error("not a npy: " + path);
    uint8_t major = 0, minor = 0; f.read((char*)&major,1); f.read((char*)&minor,1);
    uint32_t hlen = 0;
    if (major == 1) { uint16_t h; f.read((char*)&h,2); hlen = h; } else { f.read((char*)&hlen,4); }
    std::string header(hlen, '\0'); f.read(&header[0], hlen);
    if (header.find("<f4") == std::string::npos) throw std::runtime_error("npy not <f4: " + path);
    NpyF32 out;
    auto lp = header.find('('), rp = header.find(')', lp);
    std::string dims = header.substr(lp+1, rp-lp-1);
    for (size_t i = 0; i < dims.size();) {
        while (i < dims.size() && (dims[i]==' '||dims[i]==',')) i++;
        if (i >= dims.size()) break;
        int64_t v = 0; bool any = false;
        while (i < dims.size() && dims[i]>='0' && dims[i]<='9') { v=v*10+(dims[i]-'0'); i++; any=true; }
        if (any) out.shape.push_back(v);
    }
    out.data.resize(out.numel());
    f.read((char*)out.data.data(), out.numel()*sizeof(float));
    if (!f) throw std::runtime_error("short read: " + path);
    return out;
}

// ----------------------------------------------------------------- radix-2 FFT
static void fft_radix2(std::vector<std::complex<double>> & a) {
    const size_t n = a.size();
    for (size_t i = 1, j = 0; i < n; i++) {
        size_t bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    const double PI = 3.14159265358979323846;
    for (size_t len = 2; len <= n; len <<= 1) {
        std::complex<double> wl(std::cos(-2*PI/(double)len), std::sin(-2*PI/(double)len));
        for (size_t i = 0; i < n; i += len) {
            std::complex<double> w(1,0);
            for (size_t k = 0; k < len/2; k++) {
                auto u = a[i+k], v = a[i+k+len/2]*w; a[i+k]=u+v; a[i+k+len/2]=u-v; w*=wl;
            }
        }
    }
}

// ----------------------------------------------------------------- NeMo-exact log-mel -> [n_mel, T]
static std::vector<float> compute_logmel(const std::vector<float> & pcm,
        const float * window, int win_length, const float * fb, int n_mel, int n_freq,
        int n_fft, int hop, float preemph, float log_guard, int & T_out) {
    std::vector<float> x(pcm.size());
    if (!pcm.empty()) x[0] = pcm[0];
    for (size_t i = 1; i < pcm.size(); i++) x[i] = pcm[i] - preemph*pcm[i-1];
    const int pad = n_fft/2; const int64_t N = (int64_t)x.size();
    auto reflect = [&](int64_t idx)->float {
        if (N==1) return x[0]; int64_t per=2*(N-1), m=((idx%per)+per)%per; return x[m<N?m:per-m]; };
    const int T = (int)(1 + N/hop); T_out = T;
    const int woff = (n_fft - win_length)/2;
    std::vector<float> mel((size_t)n_mel*T);
    std::vector<std::complex<double>> buf(n_fft);
    for (int t = 0; t < T; t++) {
        for (int i = 0; i < n_fft; i++) buf[i] = std::complex<double>(0.0, 0.0);
        for (int j = 0; j < win_length; j++) {
            int64_t sidx = (int64_t)t*hop + woff + j - pad;
            buf[woff+j] = std::complex<double>((double)reflect(sidx)*(double)window[j], 0.0);
        }
        fft_radix2(buf);
        for (int m = 0; m < n_mel; m++) {
            double acc = 0; const float * fbm = fb + (size_t)m*n_freq;
            for (int k = 0; k < n_freq; k++) { double re=buf[k].real(),im=buf[k].imag(); acc += (double)fbm[k]*(re*re+im*im); }
            mel[(size_t)m*T + t] = (float)std::log(acc + (double)log_guard);
        }
    }
    return mel;
}

static double max_abs_diff(const float * a, const float * b, size_t n, double * mean) {
    double mx = 0, sm = 0; for (size_t i=0;i<n;i++){ double d=std::fabs((double)a[i]-(double)b[i]); mx=std::max(mx,d); sm+=d; }
    if (mean) *mean = sm/(double)std::max<size_t>(1,n); return mx;
}

int main(int argc, char ** argv) {
    std::string model = "model/sortformer.gguf";
    std::string pcm_npy = "reference/input_pcm.npy";
    std::string ref_dir = "reference_offline";
    for (int i = 1; i < argc; i++) { std::string a=argv[i]; auto nx=[&]{return (i+1<argc)?argv[++i]:"";};
        if (a=="--model") model=nx(); else if (a=="--pcm") pcm_npy=nx(); else if (a=="--ref-dir") ref_dir=nx(); }

    // ---- backend (CUDA if present) ----
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    ggml_backend_t backend = dev ? ggml_backend_dev_init(dev, nullptr) : ggml_backend_cpu_init();
    printf("backend: %s\n", dev ? ggml_backend_dev_name(dev) : "CPU");

    // ---- load all weights onto the backend ----
    struct ggml_context * ctxw = nullptr;
    struct gguf_init_params gp = { /*no_alloc*/ true, /*ctx*/ &ctxw };
    struct gguf_context * gguf = gguf_init_from_file(model.c_str(), gp);
    if (!gguf) { fprintf(stderr, "failed to load gguf\n"); return 1; }
    const int n_t = (int) gguf_get_n_tensors(gguf);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(ctxw, backend);
    {
        std::ifstream f(model, std::ios::binary);
        const size_t data_off = gguf_get_data_offset(gguf);
        std::vector<char> tmp;
        for (int i = 0; i < n_t; i++) {
            ggml_tensor * t = ggml_get_tensor(ctxw, gguf_get_tensor_name(gguf, i));
            const size_t nb = ggml_nbytes(t);
            tmp.resize(nb);
            f.seekg((std::streamoff)(data_off + gguf_get_tensor_offset(gguf, i)), std::ios::beg);
            f.read(tmp.data(), nb);
            ggml_backend_tensor_set(t, tmp.data(), 0, nb);
        }
    }
    printf("loaded %d tensors onto backend\n", n_t);
    auto W = [&](const char * n)->ggml_tensor* { ggml_tensor* t=ggml_get_tensor(ctxw,n);
        if(!t){fprintf(stderr,"missing %s\n",n);std::exit(1);} return t; };

    // ---- read window/fb back to host for the (CPU) mel ----
    ggml_tensor * t_win = W("mel.window"), * t_fb = W("mel.fb");
    const int win_length=(int)t_win->ne[0], n_freq=(int)t_fb->ne[0], n_mel=(int)t_fb->ne[1];
    std::vector<float> win(ggml_nelements(t_win)), fb(ggml_nelements(t_fb));
    ggml_backend_tensor_get(t_win, win.data(), 0, ggml_nbytes(t_win));
    ggml_backend_tensor_get(t_fb,  fb.data(),  0, ggml_nbytes(t_fb));

    // ---- mel (CPU) ----
    NpyF32 pcm = npy_load_f32(pcm_npy);
    NpyF32 mel_ref = npy_load_f32(ref_dir + "/mel.npy");
    int T = 0;
    std::vector<float> mel = compute_logmel(pcm.data, win.data(), win_length, fb.data(), n_mel, n_freq,
                                            512, 160, 0.97f, 5.960464477539063e-08f, T);
    { int Tref=(int)mel_ref.shape.back(); int Tc=std::min(T,Tref); double mx=0;
      for(int m=0;m<n_mel;m++) for(int t=1;t<Tc-1;t++)
          mx=std::max(mx,std::fabs((double)mel[(size_t)m*T+t]-(double)mel_ref.data[(size_t)m*Tref+t]));
      printf("mel: T=%d interior max|diff|=%.3e -> %s\n", T, mx, mx<5e-3?"OK":"MISMATCH"); }
    // NeMo masks the final out-of-length mel frame to 0 before the encoder; match it so the
    // conv stem sees identical input (otherwise the bad last frame corrupts the last outputs).
    for (int m = 0; m < n_mel; m++) mel[(size_t)m*T + (T-1)] = 0.0f;

    // ---- Conformer subsampling stem (dw_striding) on the backend ----
    const size_t mem = ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE*2 + ggml_graph_overhead()*2;
    struct ggml_init_params cp = { mem, nullptr, /*no_alloc*/ true };
    struct ggml_context * ctx0 = ggml_init(cp);
    ggml_cgraph * gf = ggml_new_graph(ctx0);

    ggml_tensor * inp = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, T, n_mel, 1, 1); // [T,F,1,1] == NeMo [B,1,T,F]
    ggml_set_name(inp, "mel_in"); ggml_set_input(inp);
    ggml_tensor * cur = ggml_cont(ctx0, ggml_transpose(ctx0, inp));             // [F,T,1,1]

    auto bcast = [&](ggml_tensor * b)->ggml_tensor* { return ggml_reshape_4d(ctx0, b, 1, 1, b->ne[0], 1); };
    cur = ggml_conv_2d_direct(ctx0, W("enc.pre.c0.w"), cur, 2,2,1,1,1,1);  // f32 (avoid f16 im2col)
    cur = ggml_add(ctx0, cur, bcast(W("enc.pre.c0.b")));
    cur = ggml_relu(ctx0, cur);
    cur = ggml_conv_2d_dw_direct(ctx0, W("enc.pre.c2.w"), cur, 2,2,1,1,1,1);
    cur = ggml_add(ctx0, cur, bcast(W("enc.pre.c2.b")));
    cur = ggml_conv_2d_direct(ctx0, W("enc.pre.c3.w"), cur, 1,1,0,0,1,1);
    cur = ggml_add(ctx0, cur, bcast(W("enc.pre.c3.b")));
    cur = ggml_relu(ctx0, cur);
    cur = ggml_conv_2d_dw_direct(ctx0, W("enc.pre.c5.w"), cur, 2,2,1,1,1,1);
    cur = ggml_add(ctx0, cur, bcast(W("enc.pre.c5.b")));
    cur = ggml_conv_2d_direct(ctx0, W("enc.pre.c6.w"), cur, 1,1,0,0,1,1);
    cur = ggml_add(ctx0, cur, bcast(W("enc.pre.c6.b")));
    cur = ggml_relu(ctx0, cur);
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 0, 2, 1, 3));                 // [Wf, C, Ht, 1]
    cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]);        // [Wf*C, Ht]
    cur = ggml_mul_mat(ctx0, W("enc.pre.out.w"), cur);                          // [512, Tsub]
    cur = ggml_add(ctx0, cur, W("enc.pre.out.b"));
    ggml_set_name(cur, "pre_encode_out");
    ggml_build_forward_expand(gf, cur);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(alloc, gf);
    ggml_backend_tensor_set(inp, mel.data(), 0, (size_t)T*n_mel*sizeof(float)); // mel [F,T] flat == [T,F] ggml layout
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr,"compute failed\n"); return 1; }

    const int Dsub=(int)cur->ne[0], Tsub=(int)cur->ne[1];
    std::vector<float> out((size_t)Dsub*Tsub);
    ggml_backend_tensor_get(cur, out.data(), 0, out.size()*sizeof(float));
    printf("pre_encode out: [D=%d, T=%d]\n", Dsub, Tsub);

    NpyF32 pe = npy_load_f32(ref_dir + "/pre_encode.npy");                      // [1, Tref, 512]
    const int Tpe=(int)pe.shape[1], Dpe=(int)pe.shape[2];
    printf("ref pre_encode: [T=%d, D=%d]\n", Tpe, Dpe);
    if (Dsub==Dpe && Tsub==Tpe) {
        double mean=0, mx=max_abs_diff(out.data(), pe.data.data(), out.size(), &mean); // both layout t*D+d
        // per-time-frame max + interior (exclude first/last 2 frames) to separate boundary effects
        double mx_int=0; int argf=0; double mxf=0;
        for (int t=0;t<Tsub;t++){ double fm=0; for(int d=0;d<Dsub;d++) fm=std::max(fm,std::fabs((double)out[(size_t)t*Dsub+d]-(double)pe.data[(size_t)t*Dsub+d]));
            if(fm>mxf){mxf=fm;argf=t;} if(t>=2 && t<Tsub-2) mx_int=std::max(mx_int,fm); }
        double rms=0; for (size_t i=0;i<pe.data.size();i++) rms += (double)pe.data[i]*pe.data[i];
        rms = std::sqrt(rms/(double)pe.data.size());
        printf("\npre_encode vs NeMo: max|diff|=%.3e mean|diff|=%.3e ; worst frame=%d ; interior max=%.3e ; RMS=%.2f (interior rel=%.3f%%)\n",
               mx, mean, argf, mx_int, rms, 100.0*mx_int/rms);
        printf("%s\n", (mx_int/rms)<0.01 ? "STEM OK (f32 precision; boundary from mel edge)" : "STEM MISMATCH");
    } else {
        printf("\nSTEM shape mismatch (mine [%d,%d] vs ref [%d,%d])\n", Dsub,Tsub,Tpe,Dpe);
    }

    ggml_gallocr_free(alloc); ggml_free(ctx0);
    gguf_free(gguf); ggml_backend_buffer_free(wbuf); ggml_free(ctxw); ggml_backend_free(backend);
    return 0;
}
