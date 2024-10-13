import os
import tensorrt as trt

# 定义 ONNX 模型路径和保存的 TensorRT 引擎路径
ONNX_SIM_MODEL_PATH = '../weights/yolov8n.onnx'
TENSORRT_ENGINE_PATH_PY = '../weights/yolov8n_4b.engine'


def build_engine(onnx_file_path, engine_file_path, batch_size=4, flop=16):
    # 创建 TensorRT 日志器，并设置日志级别为详细模式
    trt_logger = trt.Logger(trt.Logger.VERBOSE)  # 可设置为 trt.Logger.ERROR 来减少日志信息
    builder = trt.Builder(trt_logger)

    # 显式批处理（EXPLICIT_BATCH）模式下创建网络定义
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    # 创建一个 ONNX 解析器，用于解析 ONNX 模型并填充 TensorRT 网络
    parser = trt.OnnxParser(network, trt_logger)

    # 读取 ONNX 模型并解析
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            # 如果解析失败，输出错误信息
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    print("Completed parsing ONNX file")

    # 创建构建器配置
    config = builder.create_builder_config()

    # 设置工作空间内存限制为 2GB
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

    # 检查是否支持 FP16，如果支持则启用 FP16 模式
    if builder.platform_has_fast_fp16 and flop == 16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("Export FP16 model")

    # 设置最大批处理大小（在显式批次模式下不生效，应通过输入形状来指定）
    # builder.max_batch_size = batch_size  # 这一行对于 EXPLICIT_BATCH 模式是无效的

    # 设置优化配置文件中的最小、常规和最大批次大小
    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)  # 获取输入张量
    input_shape = input_tensor.shape  # 获取输入张量的形状

    # 修改第一个维度（通常是 batch size），设置为最小、最常规和最大批次大小
    min_shape = [1, 3, 640, 640]  # 最小 batch size = 1
    opt_shape = [2, 3, 640, 640]  # 常规 batch size
    max_shape = [batch_size, 3, 640, 640]  # 最大 batch size

    profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    # 如果已有同名的引擎文件，删除它
    if os.path.isfile(engine_file_path):
        try:
            os.remove(engine_file_path)
        except Exception:
            print("Cannot remove existing file: ", engine_file_path)

    print("Creating TensorRT Engine")

    # 设置策略源为 CUBLAS 库
    config.set_tactic_sources(1 << int(trt.TacticSource.CUBLAS))

    # 构建序列化的网络（引擎）
    serialized_engine = builder.build_serialized_network(network, config)

    # 如果引擎创建失败
    if serialized_engine is None:
        print("引擎创建失败")
        return None

    # 将序列化的引擎保存到文件
    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)
    print("序列化的引擎已保存到: ", engine_file_path)

    return serialized_engine


if __name__ == "__main__":
    # 设置 batch size 为 4，开始构建 TensorRT 引擎
    build_engine(ONNX_SIM_MODEL_PATH, TENSORRT_ENGINE_PATH_PY, batch_size=4)
