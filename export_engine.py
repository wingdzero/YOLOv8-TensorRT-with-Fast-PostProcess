import tensorrt as trt
import os

# 设置文件路径
onnx_model_path = "weights/yolov8n.onnx"  # 替换为你的ONNX模型路径
engine_file_path = "yolov8n.engine"  # 输出的TensorRT引擎文件

# 创建logger
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

def build_engine(onnx_file_path, engine_file_path):
    # 创建builder, network, parser
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # 配置最大工作空间
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

    # 读取ONNX模型文件
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    # 设置FP16精度（如果硬件支持的话）
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # 构建engine
    print("Building the TensorRT engine, this may take a while...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("ERROR: Failed to build the engine!")
        return None

    # 将engine保存到文件
    with open(engine_file_path, 'wb') as f:
        f.write(serialized_engine)

    print(f"Engine built and saved to {engine_file_path}")

    return serialized_engine


def main():
    # 检查ONNX模型是否存在
    if not os.path.exists(onnx_model_path):
        print(f"ONNX model file {onnx_model_path} not found.")
        return

    # 检查是否已经生成过engine文件
    if os.path.exists(engine_file_path):
        print(f"Engine file {engine_file_path} already exists.")
        return

    # 构建TensorRT引擎
    build_engine(onnx_model_path, engine_file_path)


if __name__ == "__main__":
    main()
