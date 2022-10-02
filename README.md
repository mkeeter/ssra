# The Solid-State Register Allocator
A simple, extremely fast, reverse linear scan register allocator.

See the [detailed write-up](https://mattkeeter.com/blog/20220-xx-xx-ssra) for an in-depth explanation.

# Building
```
wasm-pack build --target no-modules; and cp pkg/{ssra.js,ssra_bg.wasm} ~/Web/blog/2022-xx-xx-ssra/
```
