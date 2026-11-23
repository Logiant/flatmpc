# g4 envs: link with the hard-float multilib (build_flags only reach the compiler).
Import("env")
env.Append(LINKFLAGS=["-mfpu=fpv4-sp-d16", "-mfloat-abi=hard"])
