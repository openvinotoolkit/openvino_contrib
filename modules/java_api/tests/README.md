## Build

1. Download test data:
```bash
git clone https://github.com/openvinotoolkit/testdata
```

2. Create an folder `native` with native libraries. Then run `gradle` to build a package and run tests:
```bash
gradle build -Prun_tests -DMODELS_PATH=/path/to/testdata -Ddevice=CPU --info
```
