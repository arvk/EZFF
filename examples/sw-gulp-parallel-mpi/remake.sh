cd /Users/akrishnamoorthy/Desktop/v3EZFF/TEST
source /Users/akrishnamoorthy/Desktop/v3EZFF/TEST/bin/activate

rm -rf /Users/akrishnamoorthy/Desktop/v3EZFF/TEST/EZFF
cp -r /Users/akrishnamoorthy/Desktop/v3EZFF/EZFF /Users/akrishnamoorthy/Desktop/v3EZFF/TEST
cd /Users/akrishnamoorthy/Desktop/v3EZFF/TEST/EZFF
/Users/akrishnamoorthy/Desktop/v3EZFF/TEST/bin/python setup.py install

cd /Users/akrishnamoorthy/Desktop/v3EZFF/TEST/sw-gulp-parallel-mpi
/Users/akrishnamoorthy/Desktop/v3EZFF/TEST/bin/python run.py

deactivate
