#include <inference_engine.hpp>

#include "openvino_java.hpp"
#include "enum_mapping.hpp"
#include "jni_common.hpp"

using namespace InferenceEngine;

/*  We don't use delete operator for native object because we don't own this object:
    no new operator has been used to allocate memory for it */
JNIEXPORT void JNICALL Java_org_intel_openvino_compatibility_Data_delete(JNIEnv *env, jobject obj, jlong addr) {}

JNIEXPORT void JNICALL Java_org_intel_openvino_compatibility_Data_setLayout(JNIEnv *env, jobject obj, jlong addr, jint layout)
{
    static const char method_name[] = "setLayout";
    try
    {
        Data *data = reinterpret_cast<Data *>(addr);
        auto it = layout_map.find(layout);

        if (it == layout_map.end())
            throw std::runtime_error("No such layout value!");

        data->setLayout(it->second);
    } catch (const std::exception &e){
        throwJavaException(env, &e, method_name);
    }
    catch (...)
    {
        throwJavaException(env, 0, method_name);
    }
}

JNIEXPORT jint JNICALL Java_org_intel_openvino_compatibility_Data_getLayout(JNIEnv *env, jobject, jlong addr)
{
    static const char method_name[] = "getLayout";
    try
    {
        Data *data = reinterpret_cast<Data *>(addr);
        Layout layout = data->getLayout();

        return find_by_value(layout_map, layout);
    } catch (const std::exception &e){
        throwJavaException(env, &e, method_name);
    }
    catch (...)
    {
        throwJavaException(env, 0, method_name);
    }
    return 0;
}

JNIEXPORT jintArray JNICALL Java_org_intel_openvino_compatibility_Data_GetDims(JNIEnv *env, jobject obj, jlong addr)
{
    static const char method_name[] = "GetDims";
    try
    {
        Data *data = reinterpret_cast<Data *>(addr);
        std::vector<size_t> size_t_dims = data->getDims();

        jintArray result = env->NewIntArray(size_t_dims.size());
        jint *arr = env->GetIntArrayElements(result, nullptr);

        for (int i = 0; i < size_t_dims.size(); ++i)
            arr[i] = size_t_dims[i];

        env->ReleaseIntArrayElements(result, arr, 0);
        return result;
    }
    catch (const std::exception &e)
    {
        throwJavaException(env, &e, method_name);
    }
    catch (...)
    {
        throwJavaException(env, 0, method_name);
    }
    return 0;
}
