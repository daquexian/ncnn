
add_executable(mxnet2ncnn mxnet2ncnn.cpp)

# add all mxnet2ncnn tool to a virtual project group
set_property(TARGET mxnet2ncnn PROPERTY FOLDER "tools/converter")
set_target_properties(mxnet2ncnn PROPERTIES LINK_FLAGS "-s FORCE_FILESYSTEM=1 -s ALLOW_MEMORY_GROWTH=1 -s MODULARIZE=1 -s 'EXPORT_NAME=\"create_mxnet2ncnn\"' -s 'EXTRA_EXPORTED_RUNTIME_METHODS=[FS,ccall,cwrap,callMain]' -s EXPORTED_FUNCTIONS=[_main]")
