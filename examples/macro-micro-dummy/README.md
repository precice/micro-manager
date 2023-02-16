# Install Dependencies

* [preCICE](https://github.com/precice/precice)
* [pyprecice](https://github.com/precice/python-bindings)
* [Micro Manager](https://github.com/precice/micro-manager)

# Run

Run the dummy macro solver by running

```bash
python3 macro_solver.py
```

Run the Micro Manager by running

```bash
micro_manager micro-manager-config.json
```

or 

```bash
python3 run_micro_manager.py
```

# Next Steps

If you want to couple any other solver against this dummy solver be sure to adjust the preCICE configuration (participant names, mesh names, data names etc.) to the needs of your solver, compare our [step-by-step guide for new adapters](https://github.com/precice/precice/wiki/Adapter-Example).
