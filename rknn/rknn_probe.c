// rknn_probe.c — RK3566 NPU 上对 codec.rknn 做单步计时探针（Phase 1a）
// 编译：aarch64-linux-android33-clang -O2 -lrknnrt -Wl,-rpath,'$ORIGIN' -o rknn_probe rknn_probe.c
// 用法：./rknn_probe codec.rknn [runs]
#include <rknn_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static void fill_random(rknn_tensor_type_t type, void *buf, size_t bytes) {
    float *f = (float *)buf;
    int32_t *i = (int32_t *)buf;
    for (size_t b = 0; b < bytes / 4; b++) {
        float v = ((float)(b % 2048)) / 2048.0f - 0.5f;
        if (type == RKNN_TENSOR_FLOAT16) f[b] = v;
        else if (type == RKNN_TENSOR_FLOAT32) f[b] = v;
        else i[b] = (int32_t)(b % 2048);
    }
}

static size_t type_size(rknn_tensor_type_t t) {
    switch (t) {
        case RKNN_TENSOR_FLOAT16: return 2;
        case RKNN_TENSOR_FLOAT32: case RKNN_TENSOR_INT32: case RKNN_TENSOR_UINT32: return 4;
        case RKNN_TENSOR_INT8: case RKNN_TENSOR_UINT8: return 1;
        case RKNN_TENSOR_INT64: return 8;
        default: return 4;
    }
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.rknn [runs]\n", argv[0]); return 1; }
    int runs = argc > 2 ? atoi(argv[2]) : 20;

    FILE *fp = fopen(argv[1], "rb");
    if (!fp) { perror("open model"); return 1; }
    fseek(fp, 0, SEEK_END); long size = ftell(fp); fseek(fp, 0, SEEK_SET);
    void *model = malloc(size);
    if (fread(model, 1, size, fp) != (size_t)size) { perror("read"); return 1; }
    fclose(fp);
    printf("model %s %.1f MB\n", argv[1], size / 1e6);

    double t0 = now_ms();
    rknn_context ctx;
    int ret = rknn_init(&ctx, model, size, 0, NULL);
    printf("rknn_init %d (%.1f ms)\n", ret, now_ms() - t0);
    if (ret < 0) return 2;

    rknn_input_output_num io;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io));
    if (ret < 0) { printf("query num fail %d\n", ret); return 2; }
    printf("inputs=%d outputs=%d\n", io.n_input, io.n_output);

    rknn_tensor_attr in_attr[io.n_input];
    memset(in_attr, 0, sizeof(in_attr));
    for (int i = 0; i < io.n_input; i++) {
        in_attr[i].index = i;
        rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &in_attr[i], sizeof(rknn_tensor_attr));
        printf("  in[%d] %s type=%d fmt=%d elems=%zu\n", i, in_attr[i].name,
               in_attr[i].type, in_attr[i].fmt, (size_t)in_attr[i].n_elems);
    }
    rknn_tensor_attr out_attr[io.n_output];
    memset(out_attr, 0, sizeof(out_attr));
    for (int i = 0; i < io.n_output; i++) {
        out_attr[i].index = i;
        rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &out_attr[i], sizeof(rknn_tensor_attr));
        printf("  out[%d] %s type=%d elems=%zu\n", i, out_attr[i].name,
               out_attr[i].type, (size_t)out_attr[i].n_elems);
    }

    // 随机输入（探针只测耗时，不看数值）
    rknn_input inputs[io.n_input];
    memset(inputs, 0, sizeof(inputs));
    void *bufs[io.n_input];
    for (int i = 0; i < io.n_input; i++) {
        size_t bytes = in_attr[i].n_elems * type_size(in_attr[i].type);
        bufs[i] = malloc(bytes ? bytes : 4);
        fill_random(in_attr[i].type, bufs[i], bytes ? bytes : 4);
        inputs[i].index = i;
        inputs[i].type = in_attr[i].type;
        inputs[i].fmt = RKNN_TENSOR_UNDEFINED;
        inputs[i].size = bytes;
        inputs[i].buf = bufs[i];
        inputs[i].pass_through = 0;
    }
    void *obuf = malloc(1 << 20);
    rknn_output outputs[io.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < io.n_output; i++) {
        outputs[i].index = i;
        outputs[i].want_float = 0;
        outputs[i].is_prealloc = 1;
        outputs[i].size = out_attr[i].n_elems * type_size(out_attr[i].type);
        outputs[i].buf = realloc(obuf, (1 << 22));  // 单缓冲足够
    }

    for (int r = 0; r < 3; r++) {  // warmup
        rknn_inputs_set(ctx, io.n_input, inputs);
        if (rknn_run(ctx, NULL) < 0) { printf("warmup run fail\n"); return 3; }
        rknn_outputs_get(ctx, io.n_output, outputs, NULL);
        rknn_outputs_release(ctx, io.n_output, outputs);
    }
    double best = 1e9, total = 0;
    for (int r = 0; r < runs; r++) {
        double a = now_ms();
        rknn_inputs_set(ctx, io.n_input, inputs);
        rknn_run(ctx, NULL);
        rknn_outputs_get(ctx, io.n_output, outputs, NULL);
        double d = now_ms() - a;
        rknn_outputs_release(ctx, io.n_output, outputs);
        total += d; if (d < best) best = d;
        printf("run %d %.2f ms\n", r, d);
    }
    printf("RKNN_PROF avg=%.2f ms best=%.2f ms (n=%d)\n", total / runs, best, runs);
    rknn_destroy(ctx);
    return 0;
}
