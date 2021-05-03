set(DIR_SRCS)

file(GLOB TMP *.h)
list(APPEND DIR_SRCS ${TMP})
file(GLOB TMP *.cu)
list(APPEND DIR_SRCS ${TMP})

add_executable(cutlass_test ${DIR_SRCS})	
