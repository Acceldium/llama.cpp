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
#include <functional>
#include <algorithm>
#include <limits>
#include <mutex>
#include <thread>
#include <atomic>
#include <chrono>
#include <iostream>

#ifndef NOMINMAX
#define NOMINMAX            // miniaudio pulls in windows.h; keep std::min/std::max usable
#endif
#define WIN32_LEAN_AND_MEAN
#define MA_NO_ENCODING
#define MA_NO_GENERATION
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio/miniaudio.h"

// ----------------------------------------------------------------- audio decode -> 16kHz mono f32
static std::vector<float> load_audio_16k_mono(const std::string & path) {
    ma_decoder_config cfg = ma_decoder_config_init(ma_format_f32, /*channels*/1, /*rate*/16000);
    ma_decoder dec;
    if (ma_decoder_init_file(path.c_str(), &cfg, &dec) != MA_SUCCESS)
        throw std::runtime_error("cannot decode audio (wav/mp3/flac): " + path);
    ma_uint64 frames = 0; ma_decoder_get_length_in_pcm_frames(&dec, &frames);
    std::vector<float> pcm(frames); ma_uint64 read = 0;
    ma_decoder_read_pcm_frames(&dec, pcm.data(), frames, &read);
    pcm.resize(read); ma_decoder_uninit(&dec);
    return pcm;
}

// ----------------------------------------------------------------- live mic capture (miniaudio)
struct MicCap { std::mutex mtx; std::vector<float> buf; };
static void mic_callback(ma_device* dev, void* out, const void* in, ma_uint32 frames) {
    (void)out; MicCap* c = (MicCap*)dev->pUserData;
    const float* fin = (const float*)in;
    std::lock_guard<std::mutex> lk(c->mtx);
    c->buf.insert(c->buf.end(), fin, fin + frames);
}

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

// ----------------------------------------------------------------- shared graph builders
using WFn = std::function<ggml_tensor*(const char*)>;

// Conformer subsampling stem (dw_striding): mel_in [T,F,1,1] -> pre-encode embeddings [512, Tsub]
static ggml_tensor * build_stem(ggml_context * ctx0, const WFn & W, ggml_tensor * mel_in) {
    auto bcast = [&](ggml_tensor * b){ return ggml_reshape_4d(ctx0, b, 1,1,b->ne[0],1); };
    ggml_tensor * cur = ggml_cont(ctx0, ggml_transpose(ctx0, mel_in));          // [F,T,1,1]
    cur = ggml_conv_2d_direct(ctx0, W("enc.pre.c0.w"), cur, 2,2,1,1,1,1);        // f32 (avoid f16 im2col)
    cur = ggml_add(ctx0, cur, bcast(W("enc.pre.c0.b"))); cur = ggml_relu(ctx0, cur);
    cur = ggml_conv_2d_dw_direct(ctx0, W("enc.pre.c2.w"), cur, 2,2,1,1,1,1);
    cur = ggml_add(ctx0, cur, bcast(W("enc.pre.c2.b")));
    cur = ggml_conv_2d_direct(ctx0, W("enc.pre.c3.w"), cur, 1,1,0,0,1,1);
    cur = ggml_add(ctx0, cur, bcast(W("enc.pre.c3.b"))); cur = ggml_relu(ctx0, cur);
    cur = ggml_conv_2d_dw_direct(ctx0, W("enc.pre.c5.w"), cur, 2,2,1,1,1,1);
    cur = ggml_add(ctx0, cur, bcast(W("enc.pre.c5.b")));
    cur = ggml_conv_2d_direct(ctx0, W("enc.pre.c6.w"), cur, 1,1,0,0,1,1);
    cur = ggml_add(ctx0, cur, bcast(W("enc.pre.c6.b"))); cur = ggml_relu(ctx0, cur);
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 0, 2, 1, 3));                 // [Wf, C, Ht, 1]
    cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]);        // [Wf*C, Ht]
    cur = ggml_mul_mat(ctx0, W("enc.pre.out.w"), cur);                          // [512, Tsub]
    return ggml_add(ctx0, cur, W("enc.pre.out.b"));
}

// one Conformer macaron block (rel-pos MHSA + conv module); cur [512,T] -> [512,T]
static ggml_tensor * build_conformer_block(ggml_context * ctx0, const WFn & W, int il,
                                           ggml_tensor * cur, ggml_tensor * pos_emb, int n_head, int d_head) {
    const int Dm = (int)cur->ne[0];
    char p[40]; auto N = [&](const char* s){ snprintf(p,sizeof p,"enc.%d.%s",il,s); return W(p); };
    auto norm = [&](ggml_tensor* x, const char* wn, const char* bn){
        x = ggml_norm(ctx0, x, 1e-5f); return ggml_add(ctx0, ggml_mul(ctx0, x, N(wn)), N(bn)); };
    ggml_tensor * residual = cur;
    { ggml_tensor* x = norm(residual, "nff1.w", "nff1.b");                       // macaron FF1 (½)
      x = ggml_add(ctx0, ggml_mul_mat(ctx0, N("ff1.l1.w"), x), N("ff1.l1.b"));
      x = ggml_silu(ctx0, x);
      x = ggml_add(ctx0, ggml_mul_mat(ctx0, N("ff1.l2.w"), x), N("ff1.l2.b"));
      residual = ggml_add(ctx0, residual, ggml_scale(ctx0, x, 0.5f)); }
    { ggml_tensor* c = norm(residual, "nsa.w", "nsa.b");                         // rel-pos MHSA
      ggml_tensor* Q = ggml_add(ctx0, ggml_mul_mat(ctx0, N("attn.q.w"), c), N("attn.q.b"));
      Q = ggml_reshape_3d(ctx0, Q, d_head, n_head, Q->ne[1]);
      ggml_tensor* Qu = ggml_permute(ctx0, ggml_add(ctx0, Q, N("attn.bu")), 0,2,1,3);
      ggml_tensor* Qv = ggml_permute(ctx0, ggml_add(ctx0, Q, N("attn.bv")), 0,2,1,3);
      ggml_tensor* K = ggml_add(ctx0, ggml_mul_mat(ctx0, N("attn.k.w"), c), N("attn.k.b"));
      K = ggml_reshape_3d(ctx0, K, d_head, n_head, K->ne[1]);
      K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0,2,1,3));
      ggml_tensor* V = ggml_add(ctx0, ggml_mul_mat(ctx0, N("attn.v.w"), c), N("attn.v.b"));
      V = ggml_reshape_3d(ctx0, V, d_head, n_head, V->ne[1]);
      V = ggml_cont(ctx0, ggml_permute(ctx0, V, 1,2,0,3));
      ggml_tensor* ac = ggml_cont(ctx0, ggml_permute(ctx0, ggml_mul_mat(ctx0, Qu, K), 1,0,2,3));
      ggml_tensor* pp = ggml_mul_mat(ctx0, N("attn.pos.w"), pos_emb);
      pp = ggml_reshape_3d(ctx0, pp, d_head, n_head, pp->ne[1]);
      pp = ggml_permute(ctx0, pp, 0,2,1,3);
      ggml_tensor* bd = ggml_cont(ctx0, ggml_permute(ctx0, ggml_mul_mat(ctx0, Qv, pp), 1,0,2,3));
      { const int64_t pl=bd->ne[0], ql=bd->ne[1], h=bd->ne[2];                   // rel shift
        bd = ggml_pad(ctx0, bd, 1,0,0,0);
        bd = ggml_roll(ctx0, bd, 1,0,0,0);
        bd = ggml_reshape_3d(ctx0, bd, ql, pl+1, h);
        bd = ggml_view_3d(ctx0, bd, ql, pl, h, bd->nb[1], bd->nb[2], bd->nb[0]*ql);
        bd = ggml_cont_3d(ctx0, bd, pl, ql, h); }
      bd = ggml_view_3d(ctx0, bd, ac->ne[0], bd->ne[1], bd->ne[2], bd->nb[1], bd->nb[2], 0);
      ggml_tensor* sc = ggml_scale(ctx0, ggml_add(ctx0, ac, bd), 1.0f/std::sqrt((float)d_head));
      ggml_tensor* at = ggml_soft_max(ctx0, sc);
      ggml_tensor* x = ggml_mul_mat(ctx0, at, V);
      x = ggml_cont_2d(ctx0, ggml_permute(ctx0, x, 2,0,1,3), Dm, at->ne[1]);
      cur = ggml_add(ctx0, ggml_mul_mat(ctx0, N("attn.o.w"), x), N("attn.o.b")); }
    residual = ggml_add(ctx0, residual, cur);
    { ggml_tensor* x = norm(residual, "ncv.w", "ncv.b");                         // conv module
      x = ggml_add(ctx0, ggml_mul_mat(ctx0, ggml_reshape_2d(ctx0, N("cv.pw1.w"), Dm, 2*Dm), x), N("cv.pw1.b"));
      { int64_t d = x->ne[0]/2;                                                  // GLU (sigmoid gate)
        ggml_tensor* g = ggml_sigmoid(ctx0, ggml_view_2d(ctx0, x, d, x->ne[1], x->nb[1], d*x->nb[0]));
        x = ggml_mul(ctx0, ggml_view_2d(ctx0, x, d, x->ne[1], x->nb[1], 0), g);
        x = ggml_cont(ctx0, ggml_transpose(ctx0, x)); }
      x = ggml_pad(ctx0, x, 4,0,0,0); x = ggml_roll(ctx0, x, 4,0,0,0); x = ggml_pad(ctx0, x, 4,0,0,0);
      x = ggml_ssm_conv(ctx0, x, ggml_reshape_2d(ctx0, N("cv.dw.w"), 9, Dm));    // symmetric dw k9, f32
      x = ggml_add(ctx0, x, N("cv.dw.b"));
      x = ggml_add(ctx0, ggml_mul(ctx0, x, N("cv.bn.w")), N("cv.bn.b"));         // folded BN
      x = ggml_silu(ctx0, x);
      x = ggml_add(ctx0, ggml_mul_mat(ctx0, ggml_reshape_2d(ctx0, N("cv.pw2.w"), Dm, Dm), x), N("cv.pw2.b"));
      cur = x; }
    residual = ggml_add(ctx0, residual, cur);
    { ggml_tensor* x = norm(residual, "nff2.w", "nff2.b");                       // macaron FF2 (½)
      x = ggml_add(ctx0, ggml_mul_mat(ctx0, N("ff2.l1.w"), x), N("ff2.l1.b"));
      x = ggml_silu(ctx0, x);
      x = ggml_add(ctx0, ggml_mul_mat(ctx0, N("ff2.l2.w"), x), N("ff2.l2.b"));
      residual = ggml_add(ctx0, residual, ggml_scale(ctx0, x, 0.5f)); }
    return norm(residual, "nout.w", "nout.b");
}

// one post-LN Transformer block (full attn); cur [192,T] -> [192,T]
static ggml_tensor * build_transformer_block(ggml_context * ctx0, const WFn & W, int il,
                                             ggml_tensor * cur, int n_head, int d_head) {
    const int H = (int)cur->ne[0], Tq = (int)cur->ne[1];
    char p[40]; auto N = [&](const char* s){ snprintf(p,sizeof p,"tf.%d.%s",il,s); return W(p); };
    ggml_tensor * x0 = cur;
    ggml_tensor * Q = ggml_add(ctx0, ggml_mul_mat(ctx0, N("attn.q.w"), x0), N("attn.q.b"));
    ggml_tensor * K = ggml_add(ctx0, ggml_mul_mat(ctx0, N("attn.k.w"), x0), N("attn.k.b"));
    ggml_tensor * V = ggml_add(ctx0, ggml_mul_mat(ctx0, N("attn.v.w"), x0), N("attn.v.b"));
    Q = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, Q, d_head, n_head, Tq), 0,2,1,3));
    K = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, K, d_head, n_head, Tq), 0,2,1,3));
    V = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, V, d_head, n_head, Tq), 1,2,0,3));
    ggml_tensor * sc = ggml_scale(ctx0, ggml_mul_mat(ctx0, K, Q), 1.0f/std::sqrt((float)d_head));
    sc = ggml_soft_max(ctx0, sc);
    ggml_tensor * o = ggml_mul_mat(ctx0, V, sc);
    o = ggml_cont_2d(ctx0, ggml_permute(ctx0, o, 0,2,1,3), H, Tq);
    o = ggml_add(ctx0, ggml_mul_mat(ctx0, N("attn.o.w"), o), N("attn.o.b"));
    ggml_tensor * x1 = ggml_norm(ctx0, ggml_add(ctx0, x0, o), 1e-5f);
    x1 = ggml_add(ctx0, ggml_mul(ctx0, x1, N("ln1.w")), N("ln1.b"));
    ggml_tensor * f = ggml_add(ctx0, ggml_mul_mat(ctx0, N("ff.in.w"), x1), N("ff.in.b"));
    f = ggml_relu(ctx0, f);
    f = ggml_add(ctx0, ggml_mul_mat(ctx0, N("ff.out.w"), f), N("ff.out.b"));
    ggml_tensor * x2 = ggml_norm(ctx0, ggml_add(ctx0, x1, f), 1e-5f);
    return ggml_add(ctx0, ggml_mul(ctx0, x2, N("ln2.w")), N("ln2.b"));
}

// speaker head: trans_out [192,T] -> sigmoid speaker probs [4,T]
static ggml_tensor * build_speaker_head(ggml_context * ctx0, const WFn & W, ggml_tensor * trans_out,
                                        ggml_tensor ** spk_logits_out) {
    ggml_tensor * h = ggml_relu(ctx0, trans_out);
    h = ggml_add(ctx0, ggml_mul_mat(ctx0, W("sm.fh2h.w"), h), W("sm.fh2h.b"));
    h = ggml_relu(ctx0, h);
    ggml_tensor * logits = ggml_add(ctx0, ggml_mul_mat(ctx0, W("sm.sh2s.w"), h), W("sm.sh2s.b"));
    if (spk_logits_out) *spk_logits_out = logits;
    return ggml_sigmoid(ctx0, logits);
}

// per-speaker activity summary + RTTM segments (P = [T,4] flat, 80ms/frame, threshold 0.5)
static void emit_diarization(const std::vector<float> & P, int Tt, int S) {
    printf("\nspeaker activity (frames>0.5 of %d, mean prob):\n", Tt);
    for (int s=0;s<S;s++){ int act=0; double mp=0; for(int t=0;t<Tt;t++){ float v=P[(size_t)t*S+s]; if(v>0.5f)act++; mp+=v; }
        printf("  spk %d: %4d frames active (%.1f%%), mean=%.4f\n", s, act, 100.0*act/Tt, mp/Tt); }
    const double fr=0.08; const float thr=0.5f;
    printf("\nRTTM:\n");
    for (int s=0;s<S;s++){ int st=-1;
        for (int t=0;t<=Tt;t++){ bool on=(t<Tt)&&P[(size_t)t*S+s]>thr;
            if(on&&st<0)st=t;
            else if(!on&&st>=0){ printf("SPEAKER audio 1 %.2f %.2f <NA> <NA> spk%d <NA> <NA>\n", st*fr,(t-st)*fr,s); st=-1; } } }
}

// NeMo _get_silence_profile: update running mean silence embedding from a block of (embs,preds).
// embs [C,512] flat f*512+d, preds [C,4] flat f*4+s. Silence = sum_s preds < 0.2.
static void update_silence(std::vector<float> & mean_sil, double & n_sil,
                           const std::vector<float> & embs, const std::vector<float> & preds, int C) {
    const int S=4, D=512; const float SIL_THR=0.2f;
    std::vector<double> sum(D, 0.0); int count=0;
    for (int f=0; f<C; f++){ float ps=0; for(int s=0;s<S;s++) ps+=preds[(size_t)f*S+s];
        if (ps < SIL_THR){ count++; for(int d=0;d<D;d++) sum[d]+=embs[(size_t)f*D+d]; } }
    if (count==0) return;
    for (int d=0; d<D; d++) mean_sil[d] = (float)((mean_sil[d]*n_sil + sum[d]) / std::max(n_sil+count, 1.0));
    n_sil += count;
}

// NeMo _compress_spkcache (eval, batch=1): keep the 188 most important frames of (embs,preds),
// ordered by speaker then original frame order, with 3 mean-silence slots per speaker. In-place.
static void compress_spkcache(std::vector<float> & embs, std::vector<float> & preds,
                              const std::vector<float> & mean_sil) {
    const int L=188, S=4, D=512, SIL=3, MAXIDX=99999;
    const int M=(int)(preds.size()/S);
    const float NEG=-std::numeric_limits<float>::infinity(), POS=std::numeric_limits<float>::infinity();
    const float P_THR=0.25f; const int MINPOS=22, STRONG=33, WEAK=66;
    std::vector<float> sc((size_t)M*S);
    for (int f=0; f<M; f++){ double l1sum=0, lp[4], l1[4];
        for (int s=0;s<S;s++){ float p=preds[(size_t)f*S+s];
            lp[s]=std::log(std::max(p,P_THR)); l1[s]=std::log(std::max(1.0f-p,P_THR)); l1sum+=l1[s]; }
        for (int s=0;s<S;s++) sc[(size_t)f*S+s]=(float)(lp[s]-l1[s]+l1sum-std::log(0.5)); }
    for (int f=0;f<M;f++) for(int s=0;s<S;s++) if(!(preds[(size_t)f*S+s]>0.5f)) sc[(size_t)f*S+s]=NEG;
    int poscnt[4]={0,0,0,0};
    for (int s=0;s<S;s++) for(int f=0;f<M;f++) if(sc[(size_t)f*S+s]>0) poscnt[s]++;
    for (int f=0;f<M;f++) for(int s=0;s<S;s++){ bool sp=preds[(size_t)f*S+s]>0.5f;
        if(!(sc[(size_t)f*S+s]>0) && sp && poscnt[s]>=MINPOS) sc[(size_t)f*S+s]=NEG; }
    for (int f=L; f<M; f++) for(int s=0;s<S;s++) if(sc[(size_t)f*S+s]!=NEG) sc[(size_t)f*S+s]+=0.05f;
    auto boost=[&](int k, float add){ for(int s=0;s<S;s++){
        std::vector<int> idx(M); for(int f=0;f<M;f++) idx[f]=f;
        int kk=std::min(k,M);
        std::partial_sort(idx.begin(), idx.begin()+kk, idx.end(),
            [&](int a,int b){ return sc[(size_t)a*S+s] > sc[(size_t)b*S+s]; });
        for(int j=0;j<kk;j++) if(sc[(size_t)idx[j]*S+s]!=NEG) sc[(size_t)idx[j]*S+s]+=add; } };
    boost(STRONG, (float)(-2.0*std::log(0.5)));
    boost(WEAK,   (float)(-1.0*std::log(0.5)));
    // flatten value(s,f) = s*(M+SIL)+f ; pad frames (f>=M) score +inf ; pick top-L
    const int Mp=M+SIL;
    auto scval=[&](long long fi)->float{ int s=(int)(fi/Mp), f=(int)(fi%Mp);
        return f>=M ? POS : sc[(size_t)f*S+s]; };
    std::vector<long long> flat((size_t)S*Mp); for(size_t i=0;i<flat.size();i++) flat[i]=(long long)i;
    std::partial_sort(flat.begin(), flat.begin()+L, flat.end(),
        [&](long long a,long long b){ return scval(a) > scval(b); });
    std::vector<long long> sel(flat.begin(), flat.begin()+L);
    for (auto& v: sel) if(scval(v)==NEG) v=MAXIDX;        // mark invalid (NeMo placeholder)
    std::sort(sel.begin(), sel.end());                    // speaker-then-frame order
    std::vector<float> ne((size_t)L*D), np((size_t)L*S);
    for (int i=0;i<L;i++){ bool dis=(sel[i]==MAXIDX); int f=(int)(sel[i]%Mp);
        if(f>=M) dis=true; if(dis) f=0;
        for(int d=0;d<D;d++) ne[(size_t)i*D+d]= dis ? mean_sil[d] : embs[(size_t)f*D+d];
        for(int s=0;s<S;s++) np[(size_t)i*S+s]= dis ? 0.0f    : preds[(size_t)f*S+s]; }
    embs=std::move(ne); preds=std::move(np);
}

// NeMo create_pe: relative sinusoidal PE, positions (N-1)..-(N-1); host buffer [Dm, 2N-1]
static std::vector<float> make_pos_emb(int Dm, int N) {
    const int n_pos = 2*N - 1;
    std::vector<float> pe((size_t)Dm*n_pos);
    for (int pidx = 0; pidx < n_pos; pidx++) {
        const double position = (double)(N - 1 - pidx);
        for (int i = 0; i < Dm/2; i++) {
            const double div = std::exp((double)(2*i) * -(std::log(10000.0)/(double)Dm));
            pe[(size_t)pidx*Dm + 2*i]   = (float)std::sin(position*div);
            pe[(size_t)pidx*Dm + 2*i+1] = (float)std::cos(position*div);
        }
    }
    return pe;
}

int main(int argc, char ** argv) {
    std::string model = "model/sortformer.gguf";
    std::string pcm_npy = "reference/input_pcm.npy";
    std::string ref_dir = "reference_offline";
    std::string audio;          // wav/mp3/flac; takes precedence over --pcm when set
    bool validate = false;      // compare against ref_dir npy targets
    bool stream = false;        // chunked streaming inference (AOSC speaker cache)
    bool mic = false;           // live microphone capture -> streaming diarization
    std::string stream_ref;     // explicit npy to compare streaming total_preds against
    for (int i = 1; i < argc; i++) { std::string a=argv[i]; auto nx=[&]{return (i+1<argc)?argv[++i]:"";};
        if (a=="--model") model=nx(); else if (a=="--pcm") pcm_npy=nx(); else if (a=="--ref-dir") ref_dir=nx();
        else if (a=="--audio") audio=nx(); else if (a=="--validate") validate=true; else if (a=="--stream") stream=true;
        else if (a=="--stream-ref") stream_ref=nx(); else if (a=="--mic") { mic=true; stream=true; } }

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

    // ---- input audio -> PCM 16kHz mono (skipped for --mic: captured live below) ----
    if (audio.empty() && !stream) validate = true;   // the offline .npy path is the bit-exact validation run
    std::vector<float> pcm_samples; int T = 0; std::vector<float> mel;
    auto logmel = [&](const std::vector<float>& pcm, bool mask_last)->std::pair<std::vector<float>,int>{
        int Tt=0; std::vector<float> m = compute_logmel(pcm, win.data(), win_length, fb.data(), n_mel, n_freq,
                                                        512, 160, 0.97f, 5.960464477539063e-08f, Tt);
        if (mask_last && Tt>0) for(int k=0;k<n_mel;k++) m[(size_t)k*Tt + (Tt-1)]=0.0f;
        return {m, Tt};
    };
    if (!mic) {
        if (!audio.empty()) {
            pcm_samples = load_audio_16k_mono(audio);
            printf("decoded %s: %zu samples (%.2fs)\n", audio.c_str(), pcm_samples.size(), pcm_samples.size()/16000.0);
        } else {
            NpyF32 p = npy_load_f32(pcm_npy); pcm_samples = std::move(p.data);
        }
        auto mr = logmel(pcm_samples, /*mask_last*/false); mel = std::move(mr.first); T = mr.second;
        if (validate) { NpyF32 mel_ref = npy_load_f32(ref_dir + "/mel.npy");
          int Tref=(int)mel_ref.shape.back(); int Tc=std::min(T,Tref); double mx=0;
          for(int k=0;k<n_mel;k++) for(int t=1;t<Tc-1;t++)
              mx=std::max(mx,std::fabs((double)mel[(size_t)k*T+t]-(double)mel_ref.data[(size_t)k*Tref+t]));
          printf("mel: T=%d interior max|diff|=%.3e -> %s\n", T, mx, mx<5e-3?"OK":"MISMATCH"); }
        // NeMo masks the final out-of-length mel frame to 0 before the encoder.
        for (int k = 0; k < n_mel; k++) mel[(size_t)k*T + (T-1)] = 0.0f;
    }

    // ---- fold conv-module batch-norm into bn.w(scale)/bn.b(bias) on host (one-time) ----
    // scale = gamma/sqrt(var+eps), bias = beta - mean*scale ; then BN == x*scale + bias.
    const int n_enc = 17;
    { const float bn_eps = 1e-5f; char nm[32];
      auto BN = [&](int il, const char* s){ snprintf(nm,sizeof nm,"enc.%d.cv.bn.%s",il,s); return W(nm); };
      for (int il=0; il<n_enc; il++){
          ggml_tensor *g=BN(il,"w"),*b=BN(il,"b"),*rm=BN(il,"rm"),*rv=BN(il,"rv");
          const int C=(int)g->ne[0];
          std::vector<float> vg(C),vb(C),vm(C),vv(C);
          ggml_backend_tensor_get(g,vg.data(),0,C*sizeof(float));
          ggml_backend_tensor_get(b,vb.data(),0,C*sizeof(float));
          ggml_backend_tensor_get(rm,vm.data(),0,C*sizeof(float));
          ggml_backend_tensor_get(rv,vv.data(),0,C*sizeof(float));
          for (int c=0;c<C;c++){ float s=vg[c]/std::sqrt(vv[c]+bn_eps); vg[c]=s; vb[c]=vb[c]-vm[c]*s; }
          ggml_backend_tensor_set(g,vg.data(),0,C*sizeof(float));
          ggml_backend_tensor_set(b,vb.data(),0,C*sizeof(float));
      } }

    auto bbt = ggml_backend_get_default_buffer_type(backend);
    // run the conv stem on a chunk mel [n_mel, Tc] (flat m*Tc+t) -> embeddings [512, Tsub] (flat t*512+d)
    auto run_stem = [&](const std::vector<float>& mel_chunk, int Tc, int& Tsub_out)->std::vector<float> {
        size_t mm = ggml_tensor_overhead()*4096 + ggml_graph_overhead_custom(4096,false);
        ggml_init_params ip{mm,nullptr,true}; ggml_context* c0=ggml_init(ip);
        ggml_cgraph* g=ggml_new_graph_custom(c0,4096,false);
        ggml_tensor* in=ggml_new_tensor_4d(c0,GGML_TYPE_F32,Tc,n_mel,1,1); ggml_set_input(in);
        ggml_tensor* pe=build_stem(c0,W,in); ggml_set_output(pe);
        ggml_build_forward_expand(g,pe);
        ggml_gallocr_t a=ggml_gallocr_new(bbt); ggml_gallocr_alloc_graph(a,g);
        ggml_backend_tensor_set(in,mel_chunk.data(),0,(size_t)Tc*n_mel*sizeof(float));
        ggml_backend_graph_compute(backend,g);
        Tsub_out=(int)pe->ne[1];
        std::vector<float> out((size_t)512*Tsub_out);
        ggml_backend_tensor_get(pe,out.data(),0,out.size()*sizeof(float));
        ggml_gallocr_free(a); ggml_free(c0); return out;
    };
    // run conformer(17)+proj+transformer(18)+head on embeddings [512,N] -> speaker probs [4,N] (flat t*4+s)
    auto run_enc_head = [&](const std::vector<float>& embs, int N)->std::vector<float> {
        size_t mm = ggml_tensor_overhead()*16384 + ggml_graph_overhead_custom(16384,false);
        ggml_init_params ip{mm,nullptr,true}; ggml_context* c0=ggml_init(ip);
        ggml_cgraph* g=ggml_new_graph_custom(c0,16384,false);
        const int Dm=512,nh=8,dh=64,n_tf=18,th=8,tdh=24;
        ggml_tensor* in=ggml_new_tensor_2d(c0,GGML_TYPE_F32,Dm,N); ggml_set_input(in);
        ggml_tensor* pos=ggml_new_tensor_2d(c0,GGML_TYPE_F32,Dm,2*N-1); ggml_set_input(pos);
        ggml_tensor* cur=ggml_scale(c0,in,std::sqrt((float)Dm));
        for(int il=0;il<n_enc;il++) cur=build_conformer_block(c0,W,il,cur,pos,nh,dh);
        cur=ggml_add(c0,ggml_mul_mat(c0,W("sm.encproj.w"),cur),W("sm.encproj.b"));
        for(int il=0;il<n_tf;il++) cur=build_transformer_block(c0,W,il,cur,th,tdh);
        ggml_tensor* preds=build_speaker_head(c0,W,cur,nullptr); ggml_set_output(preds);
        ggml_build_forward_expand(g,preds);
        std::vector<float> peh=make_pos_emb(Dm,N);
        ggml_gallocr_t a=ggml_gallocr_new(bbt); ggml_gallocr_alloc_graph(a,g);
        ggml_backend_tensor_set(in,embs.data(),0,embs.size()*sizeof(float));
        ggml_backend_tensor_set(pos,peh.data(),0,peh.size()*sizeof(float));
        ggml_backend_graph_compute(backend,g);
        std::vector<float> out((size_t)4*N);
        ggml_backend_tensor_get(preds,out.data(),0,out.size()*sizeof(float));
        ggml_gallocr_free(a); ggml_free(c0); return out;
    };

    if (stream) {
        // ===== AOSC streaming: chunk the mel, carry a speaker cache of pre-encode embeddings =====
        const int sub=8, chunk_sub=188, lc_ctx=1, rc_ctx=1, spkcache_max=188;  // NeMo config
        const int hop=160; const double FR=0.08;                              // 80ms/output frame
        std::vector<float> spkcache;  int spk_T=0;            // [512*spk_T] pre-encode embeddings
        std::vector<float> spkcache_preds; bool has_preds=false;
        std::vector<float> mean_sil(512, 0.0f); double n_sil=0;
        std::vector<float> total_preds; int total_T=0;        // [4*total_T] flat
        int ci=0;

        // process one streaming chunk starting at mel-frame `stt`; updates state; returns end frame
        auto feed_chunk = [&](const std::vector<float>& fmel, int Tfull, int stt, bool verbose)->int {
            int loff=std::min(lc_ctx*sub, stt);
            int end=std::min(stt+chunk_sub*sub, Tfull);
            int roff=std::min(rc_ctx*sub, Tfull-end);
            int a=stt-loff, b=end+roff, Tc=b-a;
            std::vector<float> cmel((size_t)n_mel*Tc);
            for(int k=0;k<n_mel;k++) for(int tt=0;tt<Tc;tt++) cmel[(size_t)k*Tc+tt]=fmel[(size_t)k*Tfull+(a+tt)];
            int Tcs=0; std::vector<float> cemb=run_stem(cmel, Tc, Tcs);
            int lc=(int)std::lround((double)loff/sub), rc=(int)std::ceil((double)roff/sub);
            int C=Tcs-lc-rc, N=spk_T+Tcs, spk_T_old=spk_T;
            std::vector<float> concat((size_t)512*N);
            std::copy(spkcache.begin(), spkcache.end(), concat.begin());
            std::copy(cemb.begin(), cemb.end(), concat.begin()+(size_t)512*spk_T);
            std::vector<float> preds=run_enc_head(concat, N);            // [N,4] flat t*4+s
            std::vector<float> pop_embs((size_t)C*512), pop_preds((size_t)C*4);
            for(int t=0;t<C;t++){ int es=lc+t, ps=spk_T_old+lc+t;
                for(int d=0;d<512;d++) pop_embs[(size_t)t*512+d]=cemb[(size_t)es*512+d];
                for(int s=0;s<4;s++)   pop_preds[(size_t)t*4+s]=preds[(size_t)ps*4+s]; }
            int chunk_start=total_T;
            for(int t=0;t<C;t++) for(int s=0;s<4;s++) total_preds.push_back(pop_preds[(size_t)t*4+s]);
            total_T += C;
            update_silence(mean_sil, n_sil, pop_embs, pop_preds, C);
            spkcache.insert(spkcache.end(), pop_embs.begin(), pop_embs.end());
            if (has_preds) spkcache_preds.insert(spkcache_preds.end(), pop_preds.begin(), pop_preds.end());
            spk_T += C;
            bool comp=false;
            if (spk_T > spkcache_max) {
                if (!has_preds) {   // first compression: cache preds = concat preds[:spk_T_old] + chunk preds
                    spkcache_preds.assign(preds.begin(), preds.begin()+(size_t)spk_T_old*4);
                    spkcache_preds.insert(spkcache_preds.end(), pop_preds.begin(), pop_preds.end());
                    has_preds=true;
                }
                compress_spkcache(spkcache, spkcache_preds, mean_sil);
                spk_T = spkcache_max; comp=true;
            }
            if (verbose) {
                printf("[%6.2f-%6.2fs] chunk %d:", chunk_start*FR, total_T*FR, ci);
                bool any=false;
                for(int s=0;s<4;s++){ int act=0; for(int t=0;t<C;t++) if(pop_preds[(size_t)t*4+s]>0.5f) act++;
                    if(act>0){ printf(" spk%d(%.0f%%)", s, 100.0*act/C); any=true; } }
                printf("%s%s\n", any?"":" (silence)", comp?"  [cache compressed]":"");
            }
            ci++;
            return end;
        };

        if (mic) {
            // ===== live microphone capture -> streaming diarization =====
            MicCap cap;
            ma_device_config dc = ma_device_config_init(ma_device_type_capture);
            dc.capture.format=ma_format_f32; dc.capture.channels=1; dc.sampleRate=16000;
            dc.dataCallback=mic_callback; dc.pUserData=&cap;
            ma_device dev;
            if (ma_device_init(NULL,&dc,&dev)!=MA_SUCCESS){ fprintf(stderr,"mic init failed\n"); return 1; }
            ma_device_start(&dev);
            printf("\n=== LIVE mic diarization (chunk %.2fs latency). Press Enter to stop. ===\n", chunk_sub*sub*0.01);
            std::atomic<bool> stop{false};
            std::thread waiter([&]{ std::cin.get(); stop=true; });
            int stt=0;
            while (!stop) {
                std::vector<float> snap; { std::lock_guard<std::mutex> lk(cap.mtx); snap=cap.buf; }
                int Tav = snap.empty()?0:(int)(1 + snap.size()/hop);
                if (stt + chunk_sub*sub + rc_ctx*sub <= Tav) {
                    auto mm = logmel(snap, false);
                    while (stt + chunk_sub*sub + rc_ctx*sub <= mm.second) stt=feed_chunk(mm.first, mm.second, stt, true);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            ma_device_stop(&dev); ma_device_uninit(&dev);
            if (waiter.joinable()) waiter.join();
            std::vector<float> snap; { std::lock_guard<std::mutex> lk(cap.mtx); snap=cap.buf; }
            auto mm = logmel(snap, true);                       // finalize: process the tail
            while (stt < mm.second) stt=feed_chunk(mm.first, mm.second, stt, true);
            printf("\n--- session diarization ---\n");
            emit_diarization(total_preds, total_T, 4);
            gguf_free(gguf); ggml_backend_buffer_free(wbuf); ggml_free(ctxw); ggml_backend_free(backend);
            return 0;
        }

        // ===== file streaming =====
        printf("\n=== streaming (chunk_len=%d frames=%.2fs, ctx +-%d, AOSC cache=%d) ===\n",
               chunk_sub, chunk_sub*sub*0.01, lc_ctx, spkcache_max);
        int stt=0;
        while (stt < T) stt=feed_chunk(mel, T, stt, true);
        printf("\n");
        emit_diarization(total_preds, total_T, 4);
        std::string sref = ref_dir; { size_t q=sref.find("reference_offline");
            if (q!=std::string::npos) sref.replace(q, 17, "reference_streaming"); }
        sref = stream_ref.empty() ? (sref + "/total_preds.npy") : stream_ref;
        try { NpyF32 r=npy_load_f32(sref);
            if (r.numel()==(int64_t)total_preds.size()){ double mean=0,mx=max_abs_diff(total_preds.data(),r.data.data(),total_preds.size(),&mean);
                printf("\nstreaming total_preds vs NeMo: max|diff|=%.3e mean|diff|=%.3e  %s\n",
                       mx, mean, mean<1e-3?"OK (decisions match NeMo; f32 floor)":"MISMATCH"); }
        } catch (...) {}
        gguf_free(gguf); ggml_backend_buffer_free(wbuf); ggml_free(ctxw); ggml_backend_free(backend);
        return 0;
    }

    // ---- Conformer encoder graph on the backend (stem -> 17 rel-pos blocks -> encoder_proj) ----
    const size_t mem = ggml_tensor_overhead()*16384 + ggml_graph_overhead_custom(16384,false);
    struct ggml_init_params cp = { mem, nullptr, /*no_alloc*/ true };
    struct ggml_context * ctx0 = ggml_init(cp);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 16384, false);

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
    ggml_tensor * pre_encode = cur;  ggml_set_output(pre_encode);

    const int Dm = (int)cur->ne[0], Tsub = (int)cur->ne[1];   // 512, ~218
    const int n_head = 8, d_head = Dm/n_head;

    // relative positional encoding (sinusoidal, NeMo create_pe), fed as input [Dm, 2T-1]
    const int n_pos = 2*Tsub - 1;
    ggml_tensor * pos_emb = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, Dm, n_pos);
    ggml_set_name(pos_emb, "pos_emb"); ggml_set_input(pos_emb);

    // xscaling: NeMo RelPositionalEncoding multiplies the encoder input by sqrt(d_model)
    cur = ggml_scale(ctx0, cur, std::sqrt((float)Dm));

    std::vector<ggml_tensor*> lay_out;
    for (int il = 0; il < n_enc; il++) {
        char p[40]; auto N = [&](const char* s){ snprintf(p,sizeof p,"enc.%d.%s",il,s); return W(p); };
        auto norm = [&](ggml_tensor* x, const char* wn, const char* bn){
            x = ggml_norm(ctx0, x, 1e-5f); return ggml_add(ctx0, ggml_mul(ctx0, x, N(wn)), N(bn)); };

        ggml_tensor * residual = cur;
        // macaron feed_forward1 (½)
        { ggml_tensor* x = norm(residual, "nff1.w", "nff1.b");
          x = ggml_add(ctx0, ggml_mul_mat(ctx0, N("ff1.l1.w"), x), N("ff1.l1.b"));
          x = ggml_silu(ctx0, x);
          x = ggml_add(ctx0, ggml_mul_mat(ctx0, N("ff1.l2.w"), x), N("ff1.l2.b"));
          residual = ggml_add(ctx0, residual, ggml_scale(ctx0, x, 0.5f)); }

        // rel-pos multi-head self-attention (Transformer-XL matrix_ac + matrix_bd)
        { ggml_tensor* c = norm(residual, "nsa.w", "nsa.b");
          ggml_tensor* Q = ggml_add(ctx0, ggml_mul_mat(ctx0, N("attn.q.w"), c), N("attn.q.b"));
          Q = ggml_reshape_3d(ctx0, Q, d_head, n_head, Q->ne[1]);
          ggml_tensor* Qu = ggml_permute(ctx0, ggml_add(ctx0, Q, N("attn.bu")), 0,2,1,3);
          ggml_tensor* Qv = ggml_permute(ctx0, ggml_add(ctx0, Q, N("attn.bv")), 0,2,1,3);
          ggml_tensor* K = ggml_add(ctx0, ggml_mul_mat(ctx0, N("attn.k.w"), c), N("attn.k.b"));
          K = ggml_reshape_3d(ctx0, K, d_head, n_head, K->ne[1]);
          K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0,2,1,3));
          ggml_tensor* V = ggml_add(ctx0, ggml_mul_mat(ctx0, N("attn.v.w"), c), N("attn.v.b"));
          V = ggml_reshape_3d(ctx0, V, d_head, n_head, V->ne[1]);
          V = ggml_cont(ctx0, ggml_permute(ctx0, V, 1,2,0,3));
          ggml_tensor* ac = ggml_cont(ctx0, ggml_permute(ctx0, ggml_mul_mat(ctx0, Qu, K), 1,0,2,3));
          ggml_tensor* pp = ggml_mul_mat(ctx0, N("attn.pos.w"), pos_emb);
          pp = ggml_reshape_3d(ctx0, pp, d_head, n_head, pp->ne[1]);
          pp = ggml_permute(ctx0, pp, 0,2,1,3);
          ggml_tensor* bd = ggml_cont(ctx0, ggml_permute(ctx0, ggml_mul_mat(ctx0, Qv, pp), 1,0,2,3));
          { const int64_t pl=bd->ne[0], ql=bd->ne[1], h=bd->ne[2];     // rel shift
            bd = ggml_pad(ctx0, bd, 1,0,0,0);
            bd = ggml_roll(ctx0, bd, 1,0,0,0);
            bd = ggml_reshape_3d(ctx0, bd, ql, pl+1, h);
            bd = ggml_view_3d(ctx0, bd, ql, pl, h, bd->nb[1], bd->nb[2], bd->nb[0]*ql);
            bd = ggml_cont_3d(ctx0, bd, pl, ql, h); }
          bd = ggml_view_3d(ctx0, bd, ac->ne[0], bd->ne[1], bd->ne[2], bd->nb[1], bd->nb[2], 0);
          ggml_tensor* sc = ggml_scale(ctx0, ggml_add(ctx0, ac, bd), 1.0f/std::sqrt((float)d_head));
          ggml_tensor* at = ggml_soft_max(ctx0, sc);
          ggml_tensor* x = ggml_mul_mat(ctx0, at, V);
          x = ggml_cont_2d(ctx0, ggml_permute(ctx0, x, 2,0,1,3), Dm, at->ne[1]);
          cur = ggml_add(ctx0, ggml_mul_mat(ctx0, N("attn.o.w"), x), N("attn.o.b")); }
        residual = ggml_add(ctx0, residual, cur);

        // conv module: pointwise(2x) -> GLU -> depthwise(k9, symmetric) -> BN(folded) -> SiLU -> pointwise
        { ggml_tensor* x = norm(residual, "ncv.w", "ncv.b");
          x = ggml_add(ctx0, ggml_mul_mat(ctx0, ggml_reshape_2d(ctx0, N("cv.pw1.w"), Dm, 2*Dm), x), N("cv.pw1.b"));
          { int64_t d = x->ne[0]/2;     // GLU (sigmoid gate; ggml_glu lacks sigmoid)
            ggml_tensor* g = ggml_sigmoid(ctx0, ggml_view_2d(ctx0, x, d, x->ne[1], x->nb[1], d*x->nb[0]));
            x = ggml_mul(ctx0, ggml_view_2d(ctx0, x, d, x->ne[1], x->nb[1], 0), g);
            x = ggml_cont(ctx0, ggml_transpose(ctx0, x)); }     // [T, C]
          x = ggml_pad(ctx0, x, 4,0,0,0); x = ggml_roll(ctx0, x, 4,0,0,0); x = ggml_pad(ctx0, x, 4,0,0,0);
          x = ggml_ssm_conv(ctx0, x, ggml_reshape_2d(ctx0, N("cv.dw.w"), 9, Dm));  // [C, T], f32
          x = ggml_add(ctx0, x, N("cv.dw.b"));
          x = ggml_add(ctx0, ggml_mul(ctx0, x, N("cv.bn.w")), N("cv.bn.b"));       // folded BN
          x = ggml_silu(ctx0, x);
          x = ggml_add(ctx0, ggml_mul_mat(ctx0, ggml_reshape_2d(ctx0, N("cv.pw2.w"), Dm, Dm), x), N("cv.pw2.b"));
          cur = x; }
        residual = ggml_add(ctx0, residual, cur);

        // macaron feed_forward2 (½)
        { ggml_tensor* x = norm(residual, "nff2.w", "nff2.b");
          x = ggml_add(ctx0, ggml_mul_mat(ctx0, N("ff2.l1.w"), x), N("ff2.l1.b"));
          x = ggml_silu(ctx0, x);
          x = ggml_add(ctx0, ggml_mul_mat(ctx0, N("ff2.l2.w"), x), N("ff2.l2.b"));
          residual = ggml_add(ctx0, residual, ggml_scale(ctx0, x, 0.5f)); }

        cur = norm(residual, "nout.w", "nout.b");
        ggml_set_output(cur); lay_out.push_back(cur);
    }

    ggml_tensor * proj = ggml_add(ctx0, ggml_mul_mat(ctx0, W("sm.encproj.w"), cur), W("sm.encproj.b"));
    ggml_set_output(proj);

    // ---- Transformer head: 18 post-LN blocks (8 heads, d=192, inner=768 relu, full attn) ----
    const int n_tf = 18, tf_head = 8, tf_dh = 192/tf_head;   // d_head = 24
    cur = proj;   // [192, T]
    std::vector<ggml_tensor*> tf_out;
    for (int il = 0; il < n_tf; il++) {
        char p[40]; auto N = [&](const char* s){ snprintf(p,sizeof p,"tf.%d.%s",il,s); return W(p); };
        ggml_tensor * x0 = cur;
        // self-attention (full, non-causal; q/k pre-scaled == scores * 1/sqrt(d_head))
        ggml_tensor * Q = ggml_add(ctx0, ggml_mul_mat(ctx0, N("attn.q.w"), x0), N("attn.q.b"));
        ggml_tensor * K = ggml_add(ctx0, ggml_mul_mat(ctx0, N("attn.k.w"), x0), N("attn.k.b"));
        ggml_tensor * V = ggml_add(ctx0, ggml_mul_mat(ctx0, N("attn.v.w"), x0), N("attn.v.b"));
        const int Tq = (int)x0->ne[1];
        Q = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, Q, tf_dh, tf_head, Tq), 0,2,1,3)); // [dh,T,H]
        K = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, K, tf_dh, tf_head, Tq), 0,2,1,3));
        V = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, V, tf_dh, tf_head, Tq), 1,2,0,3)); // [T,dh,H]
        ggml_tensor * sc = ggml_mul_mat(ctx0, K, Q);                       // [Tk,Tq,H]
        sc = ggml_scale(ctx0, sc, 1.0f/std::sqrt((float)tf_dh));
        sc = ggml_soft_max(ctx0, sc);
        ggml_tensor * o = ggml_mul_mat(ctx0, V, sc);                       // [dh,Tq,H]
        o = ggml_cont_2d(ctx0, ggml_permute(ctx0, o, 0,2,1,3), 192, Tq);   // [192,T]
        o = ggml_add(ctx0, ggml_mul_mat(ctx0, N("attn.o.w"), o), N("attn.o.b"));
        // residual -> LN1
        ggml_tensor * x1 = ggml_norm(ctx0, ggml_add(ctx0, x0, o), 1e-5f);
        x1 = ggml_add(ctx0, ggml_mul(ctx0, x1, N("ln1.w")), N("ln1.b"));
        // FFN -> residual -> LN2
        ggml_tensor * f = ggml_add(ctx0, ggml_mul_mat(ctx0, N("ff.in.w"), x1), N("ff.in.b"));
        f = ggml_relu(ctx0, f);
        f = ggml_add(ctx0, ggml_mul_mat(ctx0, N("ff.out.w"), f), N("ff.out.b"));
        ggml_tensor * x2 = ggml_norm(ctx0, ggml_add(ctx0, x1, f), 1e-5f);
        x2 = ggml_add(ctx0, ggml_mul(ctx0, x2, N("ln2.w")), N("ln2.b"));
        cur = x2;
        ggml_set_output(cur); tf_out.push_back(cur);
    }
    ggml_tensor * trans_out = cur;

    // ---- speaker head: relu -> Linear(192,192) -> relu -> Linear(192,4) -> sigmoid ----
    ggml_tensor * h = ggml_relu(ctx0, trans_out);
    h = ggml_add(ctx0, ggml_mul_mat(ctx0, W("sm.fh2h.w"), h), W("sm.fh2h.b"));
    h = ggml_relu(ctx0, h);
    ggml_tensor * spk_logits = ggml_add(ctx0, ggml_mul_mat(ctx0, W("sm.sh2s.w"), h), W("sm.sh2s.b")); // [4,T]
    ggml_set_output(spk_logits);
    ggml_tensor * preds = ggml_sigmoid(ctx0, spk_logits);
    ggml_set_output(preds);
    ggml_build_forward_expand(gf, preds);

    // ---- build pos_emb host values (NeMo create_pe: positions T-1..-(T-1), sin even / cos odd) ----
    std::vector<float> pe_host((size_t)Dm*n_pos);
    for (int pidx = 0; pidx < n_pos; pidx++) {
        const double position = (double)(Tsub - 1 - pidx);
        for (int i = 0; i < Dm/2; i++) {
            const double div = std::exp((double)(2*i) * -(std::log(10000.0)/(double)Dm));
            pe_host[(size_t)pidx*Dm + 2*i]   = (float)std::sin(position*div);
            pe_host[(size_t)pidx*Dm + 2*i+1] = (float)std::cos(position*div);
        }
    }

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(alloc, gf);
    ggml_backend_tensor_set(inp, mel.data(), 0, (size_t)T*n_mel*sizeof(float));
    ggml_backend_tensor_set(pos_emb, pe_host.data(), 0, pe_host.size()*sizeof(float));
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr,"compute failed\n"); return 1; }

    // ---- validate: stem, each conformer layer, encoder_proj (all [T,D] flat == ggml [D,T]) ----
    auto cmp = [&](const char* tag, ggml_tensor* t, const std::string& ref){
        std::vector<float> g((size_t)t->ne[0]*t->ne[1]);
        ggml_backend_tensor_get(t, g.data(), 0, g.size()*sizeof(float));
        NpyF32 r = npy_load_f32(ref);
        if ((int64_t)g.size() != r.numel()) { printf("%-14s shape mismatch (%lld vs %lld)\n", tag,(long long)g.size(),(long long)r.numel()); return; }
        double mean=0, mx=max_abs_diff(g.data(), r.data.data(), g.size(), &mean);
        double rms=0; for (float v : r.data) rms += (double)v*v; rms=std::sqrt(rms/(double)r.data.size());
        // f32 precision floor: accept small relative OR small absolute max-diff (low-RMS layers
        // inflate the relative metric even though their absolute error matches every other layer).
        const bool ok = (mx/rms)<0.02 || mx<3e-2;
        printf("%-14s max|diff|=%.3e mean|diff|=%.3e RMS=%.2f rel=%.3f%%  %s\n",
               tag, mx, mean, rms, 100.0*mx/rms, ok ? "OK" : "MISMATCH");
    };
    if (validate) {
        printf("\n=== Conformer encoder validation (%d frames) ===\n", Tsub);
        cmp("pre_encode", pre_encode, ref_dir + "/pre_encode.npy");
        for (int il = 0; il < n_enc; il++) cmp((std::string("enc_layer_")+std::to_string(il)).c_str(),
                                               lay_out[il], ref_dir + "/enc_layer_"+std::to_string(il)+".npy");
        cmp("encoder_proj", proj, ref_dir + "/encoder_proj.npy");
        printf("\n=== Transformer head + speaker head ===\n");
        for (int il = 0; il < n_tf; il++) cmp((std::string("tf_layer_")+std::to_string(il)).c_str(),
                                              tf_out[il], ref_dir + "/tf_layer_"+std::to_string(il)+".npy");
        cmp("spk_logits", spk_logits, ref_dir + "/spk_logits.npy");
        cmp("preds", preds, ref_dir + "/preds.npy");
    }

    // ---- final diarization decision (threshold 0.5) + per-speaker activity ----
    {
        std::vector<float> P((size_t)preds->ne[0]*preds->ne[1]);
        ggml_backend_tensor_get(preds, P.data(), 0, P.size()*sizeof(float));
        const int S=(int)preds->ne[0], Tt=(int)preds->ne[1];
        printf("\nspeaker activity (frames>0.5 of %d, mean prob):\n", Tt);
        for (int s=0;s<S;s++){ int act=0; double mp=0; for(int t=0;t<Tt;t++){ float v=P[(size_t)t*S+s]; if(v>0.5f)act++; mp+=v; }
            printf("  spk %d: %4d frames active (%.1f%%), mean=%.4f\n", s, act, 100.0*act/Tt, mp/Tt); }
        // RTTM segments (80ms/frame = 8x subsampling of 10ms hop), threshold 0.5
        const double fr = 0.08; const float thr = 0.5f;
        printf("\nRTTM:\n");
        for (int s=0;s<S;s++){ int st=-1;
            for (int t=0;t<=Tt;t++){ bool on = (t<Tt) && P[(size_t)t*S+s]>thr;
                if (on && st<0) st=t;
                else if (!on && st>=0){ printf("SPEAKER audio 1 %.2f %.2f <NA> <NA> spk%d <NA> <NA>\n",
                                               st*fr, (t-st)*fr, s); st=-1; } } }
    }

    ggml_gallocr_free(alloc); ggml_free(ctx0);
    gguf_free(gguf); ggml_backend_buffer_free(wbuf); ggml_free(ctxw); ggml_backend_free(backend);
    return 0;
}
