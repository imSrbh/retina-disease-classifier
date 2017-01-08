## About

This project implements algorithm for retinal diseases (Age-related macular degeneration, Diabetic retinopathy) classification based on images of the patient's retina (fundus images). 

## Installation

### 1. CMake, version 3.3 or later:
```
wget http://www.cmake.org/files/v3.4/cmake-3.4.1.tar.gz
tar -xvzf cmake-3.4.1.tar.gz
cd cmake-3.4.1/
./configure
make
sudo make install
sudo update-alternatives --install /usr/bin/cmake cmake /usr/local/bin/cmake 1 --force
```
### 2. Compilation of source code:
```
mkdir build
cd build
cmake ../
make
```
## Usage manual

```
Usage: ./retinaDiseaseClasifier [ OPTIONS ... ] IMAGE

 -h       	Prints usage manual to stdout.
 -o PATH  	Save image with marked findings in it ti file.
 -m PATH  	Set path to statistical models for clasificator.
            Mandatory in combination with '-t'.
 -d PATH  	Set path to database files of property vectors of classified findings.
            Mandatory in combination with '-c' and '-t'. 
 -c [0-2] 	Manual classification of findings. Value specifies type of desired specifies: 0 = drusen, 1 = exudates, 2 = hemorrhages.
 -t [0-2] 	Reset classificator by new database. Value specifies type of desired specifies: 0 = drusen, 1 = exudates, 2 = hemorrhages.
 ```
 
## License

This content is released under the (http://opensource.org/licenses/MIT) MIT License.
