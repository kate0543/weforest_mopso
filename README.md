# Weighted Deep Forest

> Apply variable-length PSO to find the optimal structure of Weighted Random Forest

## Related Projects

- [gcForest official source code](https://github.com/kingfengji/gcForest) 

## Install

```
```

## Usage
- Create file `data_helper.py`:
```
data_folder = 'path/to/data/folder'
file_list = ['abalone', 'balance',...]
```

- Run:
    - __VLGA__:
    ```
    python demo_weighted_gcforest_vlga.py <from_file_id> <to_file_id>
    ```
    - __VLPSO__:
    ```
    python demo_weighted_gcforest_vlpso.py <from_file_id> <to_file_id>
    ```
    - __RandomForest__:
    ```
    python demo_randomforest.py <from_file_id> <to_file_id>
    ```

## License

