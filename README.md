# The Solid-State Register Allocator
A simple, extremely fast, reverse linear scan register allocator.

See the [detailed write-up](https://mattkeeter.com/blog/2022-10-04-ssra) for an in-depth explanation.

# Building
```
wasm-pack build --target no-modules; and cp pkg/{ssra.js,ssra_bg.wasm} ~/Web/blog/2022-10-04-ssra/
```

# Author's note
Through [parallel evolution](https://en.wikipedia.org/wiki/Parallel_evolution),
it [turns out](https://news.ycombinator.com/item?id=33093358)
that this code has the same design as the LuaJIT register allocator;
if you'd like to learn more about their implementation, see
[Mike Pall's IP disclosure](https://lua-users.org/lists/lua-l/2009-11/msg00089.html)
– disregarding the SSL error – or their
[source code](https://github.com/LuaJIT/LuaJIT/blob/5e3c45c43bb0e0f1f2917d432e9d2dba12c42a6e/src/lj_asm.c#L198).
