## IO

This module offers some methods for data load/dump and file operations.

### Data load/dump
`cvbase` provides a universal api for loading and dumping data, currently
supported formats are json, yaml and pickle.

```python
import cvbase as cvb

# load data from a file
data = cvb.load('test.json')
data = cvb.load('test.yaml')
data = cvb.load('test.pickle')
# load data from a file-like object
with open('test.json', 'r') as f:
    data = cvb.load(f)

# dump data to a string
json_str = cvb.dump(data, format='json')
# dump data to a file with a filename (infer format from file extension)
cvb.dump(data, 'out.pickle')
# dump data to a file with a file-like object
with open('test.yaml', 'w') as f:
    data = cvb.dump(data, f, format='yaml')
```

### Load list from a text file

For example a.txt is a text file with 5 lines.
```
a
b
c
d
e
```

Then use `list_from_file` to load the list from a.txt.

```python
import cvbase as cvb

cvb.list_from_file('a.txt')
# output ['a', 'b', 'c', 'd', 'e']
cvb.list_from_file('a.txt', offset=2)
# output ['c', 'd', 'e']
cvb.list_from_file('a.txt', max_num=2)
# output ['a', 'b']
cvb.list_from_file('a.txt', prefix='/mnt/')
# output ['/mnt/a', '/mnt/b', '/mnt/c', '/mnt/d', '/mnt/e']
```

### File/Directory operations

Use `check_file_exist` to check if a file exists, if not, a `FileNotFoundError`
or `IOError` will be thrown out.

Use `mkdir_or_exist` to check if a directory exists, the directory will be created if not exists.

Use `scandir` to scan a directory for all files or files will certain suffix.

```python
import cvbase as cvb

# scan the folder "test" for all files 
for filename in cvb.scandir('test'):
    print(filename)
# scan the folder "test" for all jpg files 
for filename in cvb.scandir('test', '.jpg'):
    print(filename)
```