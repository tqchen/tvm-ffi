import gdb, re, sys, os
sym = "_ZN9tvm_hooks24BinaryMaybeInplaceMutateIN3tvm4prim7AddNodeEEE9TVMFFIAnyPNS1_3ffi20StructuralMutatorObjENS5_7AnyViewE"
out = os.environ["TRACE_OUT"]
gdb.execute("set pagination off")
gdb.execute("set confirm off")
gdb.execute("starti")
vt = int(gdb.parse_and_eval("(long)&'tvm::ffi::bench::MinimalMutatorObj::VTable()::vtable'"))
bp = gdb.Breakpoint("*" + sym, internal=True)
bp.condition = "*(long*)($rdi+0x18) == %d" % vt
gdb.execute("continue")
lo = int(gdb.parse_and_eval("(long)&%s" % sym))
# function size from nm is passed in
hi = lo + int(os.environ["TRACE_SIZE"], 16)
n = calls = locks = 0
lines = []
while True:
    pc = int(gdb.parse_and_eval("$pc"))
    if not (lo <= pc < hi):
        break
    ins = gdb.execute("x/i $pc", to_string=True).strip()
    text = ins.split(":", 1)[1].strip() if ":" in ins else ins
    lines.append("+0x%x: %s" % (pc - lo, text))
    n += 1
    if text.startswith("call"): calls += 1
    if text.startswith("lock"): locks += 1
    gdb.execute("ni", to_string=True)
with open(out, "w") as f:
    f.write("\n".join(lines) + "\n")
    f.write("# executed %d instructions, %d calls, %d lock-prefixed\n" % (n, calls, locks))
print("executed %d instructions, %d calls, %d lock-prefixed" % (n, calls, locks))
gdb.execute("kill")
