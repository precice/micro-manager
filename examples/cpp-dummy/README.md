For the micro dummy: In `cpp-dummy/` run `g++ -std=c++11 -shared -fPIC micro_dummy.cpp -o micro_dummy.so` and then in `examples/` run `python run_micro_manager.py`:

```bash
cd cpp-dummy/
g++ -std=c++11 -shared -fPIC micro_dummy.cpp -o micro_dummy.so
cd ../
python run_micro_manager.py
```



For the macro dummy: In `examples/` run

```bash
python macro_dummy.py
```