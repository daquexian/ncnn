add_executable(darknet2ncnn darknet2ncnn.cpp)
set_property(TARGET darknet2ncnn PROPERTY FOLDER "tools/converter")
set_target_properties(darknet2ncnn PROPERTIES LINK_FLAGS "-s FORCE_FILESYSTEM=1 -s ALLOW_MEMORY_GROWTH=1 -s MODULARIZE=1 -s 'EXPORT_NAME=\"create_darknet2ncnn\"' -s 'EXTRA_EXPORTED_RUNTIME_METHODS=[FS,ccall,cwrap,callMain]' -s EXPORTED_FUNCTIONS=[_main]")
