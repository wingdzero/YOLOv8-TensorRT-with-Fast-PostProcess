import os
import tensorrt as trt

ONNX_SIM_MODEL_PATH = '../weights/yolov8n_1b.onnx'
TENSORRT_ENGINE_PATH_PY = './yolov8n_1b.engine'


def build_engine(onnx_file_path, engine_file_path, flop=16):
    trt_logger = trt.Logger(trt.Logger.VERBOSE)  # trt.Logger.ERROR
    builder = trt.Builder(trt_logger)
    network = builder.create_network(
        1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )

    parser = trt.OnnxParser(network, trt_logger)
    # Parse ONNX model
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    print("Completed parsing ONNX file")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

    # Enable FP16 if supported
    if builder.platform_has_fast_fp16 and flop == 16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("Export FP16 model")

    if os.path.isfile(engine_file_path):
        try:
            os.remove(engine_file_path)
        except Exception:
            print("Cannot remove existing file: ", engine_file_path)

    print("Creating TensorRT Engine")

    config.set_tactic_sources(1 << int(trt.TacticSource.CUBLAS))

    serialized_engine = builder.build_serialized_network(network, config)

    # 打印出分析结果
    # inspector.print_layer_times()

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
    build_engine(ONNX_SIM_MODEL_PATH, TENSORRT_ENGINE_PATH_PY)
