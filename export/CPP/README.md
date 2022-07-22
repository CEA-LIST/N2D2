# N2D2 CPP Export

## Compile the export

To compile the export, run the command

```
make
```

To compile the export with the *SAVE_OUTPUTS* option activated, run

```
make save_outputs
```

## Test the export

To test the export, you can run the binary

```
./bin/run_export
```

Every stimuli present in the `stimuli/` directory will be tested to calculate the accuracy of the network. 

To test an unique stimuli

```
./bin/run_export -stimulus stimuli/env01.pgm
```

## Details about the files

The information about your model are stored in `dnn/`. In `dnn/include`, you will find the information about the layers in your model while in `dnn/src`, you will find functions to use the model.



