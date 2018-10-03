#!/bin/bash

cd ~/learnData

cd 01/
find ./ -type f | sed 's!^.*/!!' | xargs -i cp {}  ~/face2/face_image/00.{}
cd 02/
find ./ -type f | sed 's!^.*/!!' | xargs -i cp {}  ~/face2/face_image/01.{}
cd 03/
find ./ -type f | sed 's!^.*/!!' | xargs -i cp {}  ~/face2/face_image/02.{}
cd 04/
find ./ -type f | sed 's!^.*/!!' | xargs -i cp {}  ~/face2/face_image/03.{}
cd 05/
find ./ -type f | sed 's!^.*/!!' | xargs -i cp {}  ~/face2/face_image/04.{}
cd 06/
find ./ -type f | sed 's!^.*/!!' | xargs -i cp {}  ~/face2/face_image/05.{}
cd 07/
find ./ -type f | sed 's!^.*/!!' | xargs -i cp {}  ~/face2/face_image/06.{}
cd 08/
find ./ -type f | sed 's!^.*/!!' | xargs -i cp {}  ~/face2/face_image/07.{}
cd 09/
find ./ -type f | sed 's!^.*/!!' | xargs -i cp {}  ~/face2/face_image/08.{}
cd 10/
find ./ -type f | sed 's!^.*/!!' | xargs -i cp {}  ~/face2/face_image/09.{}

