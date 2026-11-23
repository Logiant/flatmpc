# energy_float: single-precision literals in the solver TU only (the bench data stays double).
Import("env")


def solver_only(env, node):
    print("MW", node.get_path())
    if "lqocp_core" in node.get_path():
        return env.Object(node, CCFLAGS=env["CCFLAGS"] + ["-fsingle-precision-constant"])
    return node


env.AddBuildMiddleware(solver_only)
