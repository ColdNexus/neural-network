function(set_compile_options TARGET)
  target_compile_options("${TARGET}" PRIVATE
    -Wall
    -Wcast-align # warn for potential performance problem casts
    -Woverloaded-virtual # warn if you overload (not override) a virtual function
    -Wpedantic # warn if non-standard C++ is used
    -Wnull-dereference # warn if a null dereference is detected
    -Wformat=2 # warn on security issues around functions that format output (i.e. printf)
  )
endfunction()
