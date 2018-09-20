#include <layer.h>
#include <layer/convolution.h>
#include <layer/batchnorm.h>
#include <net.h>

namespace ncnn {
class OpFuser {
public:
    void fuse(const char* parampath, const char* modelpath) {
        Net net;
        auto ret = net.load_param(parampath);
        if (ret != 0) {
            fprintf(stderr, "load_param failed");
            return;
        }
        ret = net.load_model(modelpath);
        if (ret != 0) {
            fprintf(stderr, "load_model failed");
            return;
        }
        for (size_t i = 0; i < net.layers.size() - 1; i++) {
            auto *layer = net.layers[i];
            auto *next_layer = net.layers[i + 1];
            fprintf(stdout, "name: %s\n", layer->name.c_str());
            auto type = layer->type;
            if (type == "Convolution") {
                Convolution *conv_layer = static_cast<Convolution *>(layer);
                auto &conv_weight = conv_layer->weight_data, &conv_bias = conv_layer->bias_data;        // [output_channel, input_channel, h, w]
                fprintf(stdout, "Convolution weight dims: %d\n", conv_weight.dims);
                if (next_layer->type == "BatchNorm") {
                    BatchNorm *bn_layer = static_cast<BatchNorm *>(next_layer);
                    auto &bn_a = bn_layer->a_data, bn_b = bn_layer->b_data;     // [channel]
                    size_t step = conv_weight.total() / conv_layer->num_output;
                    for (int n = 0; n < conv_layer->num_output; n++) {
                        for (size_t s = 0; s < step; s++) {
                            
                        }
                    }
                    fprintf(stdout, "bn a dims: %d\n", bn_a.dims);
                    fprintf(stdout, "bn b dims: %d\n", bn_b.dims);
                    fprintf(stdout, "Convolution %s and BatchNorm %s Need fuse!\n", conv_layer->name.c_str(), bn_layer->name.c_str());
                }
            }
        }
    }
};
}

int main() {
    ncnn::OpFuser fuser;
    fuser.fuse("/home/daquexian/models/ncnn/ncnn.proto", "/home/daquexian/models/ncnn/ncnn.bin");
}

static std::vector<std::string> layer_names;
static std::vector<std::string> blob_names;

static int find_blob_index_by_name(const char* name)
{
    for (std::size_t i=0; i<blob_names.size(); i++)
    {
        if (blob_names[i] == name)
        {
            return i;
        }
    }

    fprintf(stderr, "find_blob_index_by_name %s failed\n", name);
    return -1;
}

static void sanitize_name(char* name)
{
    for (std::size_t i=0; i<strlen(name); i++)
    {
        if (!isalnum(name[i]))
        {
            name[i] = '_';
        }
    }
}

static std::string path_to_varname(const char* path)
{
    const char* lastslash = strrchr(path, '/');
    const char* name = lastslash == NULL ? path : lastslash + 1;

    std::string varname = name;
    sanitize_name((char*)varname.c_str());

    return varname;
}

#if 0
static int dump_param(const char* parampath, const char* parambinpath, const char* idcpppath)
{
    FILE* fp = fopen(parampath, "rb");

    std::string param_var = path_to_varname(parampath);

    std::string include_guard_var = path_to_varname(idcpppath);

    int magic = 0;
    fscanf(fp, "%d", &magic);

    int layer_count = 0;
    int blob_count = 0;
    fscanf(fp, "%d %d", &layer_count, &blob_count);

    layer_names.resize(layer_count);
    blob_names.resize(blob_count);

    int blob_index = 0;
    for (int i=0; i<layer_count; i++)
    {
        int nscan = 0;

        char layer_type[33];
        char layer_name[257];
        int bottom_count = 0;
        int top_count = 0;
        nscan = fscanf(fp, "%32s %256s %d %d", layer_type, layer_name, &bottom_count, &top_count);
        if (nscan != 4)
        {
            continue;
        }

        sanitize_name(layer_name);

        int typeindex = ncnn::layer_to_index(layer_type);

//         layer->bottoms.resize(bottom_count);
        for (int i=0; i<bottom_count; i++)
        {
            char bottom_name[257];
            nscan = fscanf(fp, "%256s", bottom_name);
            if (nscan != 1)
            {
                continue;
            }

            sanitize_name(bottom_name);

            int bottom_blob_index = find_blob_index_by_name(bottom_name);
        }

//         layer->tops.resize(top_count);
        for (int i=0; i<top_count; i++)
        {
            char blob_name[257];
            nscan = fscanf(fp, "%256s", blob_name);
            if (nscan != 1)
            {
                continue;
            }

            sanitize_name(blob_name);

            blob_names[blob_index] = std::string(blob_name);

            blob_index++;
        }

        // dump layer specific params
        // parse each key=value pair
        int id = 0;
        while (fscanf(fp, "%d=", &id) == 1)
        {
            bool is_array = id <= -23300;

            if (is_array)
            {
                int len = 0;
                fscanf(fp, "%d", &len);

                for (int j = 0; j < len; j++)
                {
                    char vstr[16];
                    fscanf(fp, ",%15[^,\n ]", vstr);

                    bool is_float = vstr_is_float(vstr);

                    if (is_float)
                    {
                        float vf;
                        sscanf(vstr, "%f", &vf);
                        fwrite(&vf, sizeof(float), 1, mp);
                    }
                    else
                    {
                        int v;
                        sscanf(vstr, "%d", &v);
                        fwrite(&v, sizeof(int), 1, mp);
                    }
                }
            }
            else
            {
                char vstr[16];
                fscanf(fp, "%15s", vstr);

                bool is_float = vstr_is_float(vstr);

                if (is_float)
                {
                    float vf;
                    sscanf(vstr, "%f", &vf);
                    fwrite(&vf, sizeof(float), 1, mp);
                }
                else
                {
                    int v;
                    sscanf(vstr, "%d", &v);
                    fwrite(&v, sizeof(int), 1, mp);
                }
            }
        }

        int EOP = -233;
        fwrite(&EOP, sizeof(int), 1, mp);

        layer_names[i] = std::string(layer_name);
    }

    fclose(fp);

    return 0;
}
#endif
