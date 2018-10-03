#!/bin/bash

cd /mnt/mov/learingData/14org
find ./ -type f | sed 's!^.*/!!' | xargs -i cp {}  ~/face2/origin_image/13.{}
cd /mnt/mov/learingData/15org
find ./ -type f | sed 's!^.*/!!' | xargs -i cp {}  ~/face2/origin_image/14.{}
quit
exit

cd /mnt/mov/learingData
cd 11/
find ./ -type f | sed 's!^.*/!!' | xargs -i cp {}  ~/face2/origin_image/10.{}
cd /mnt/mov/learingData
cd 12/
find ./ -type f | sed 's!^.*/!!' | xargs -i cp {}  ~/face2/origin_image/11.{}
cd /mnt/mov/learingData
cd 13/
find ./ -type f | sed 's!^.*/!!' | xargs -i cp {}  ~/face2/origin_image/12.{}
cd /mnt/mov/learingData
cd 14/
find ./ -type f | sed 's!^.*/!!' | xargs -i cp {}  ~/face2/origin_image/13.{}
cd /mnt/mov/learingData
cd 15/
find ./ -type f | sed 's!^.*/!!' | xargs -i cp {}  ~/face2/origin_image/14.{}
cd /mnt/mov/learingData
cd 16/
find ./ -type f | sed 's!^.*/!!' | xargs -i cp {}  ~/face2/origin_image/15.{}
cd /mnt/mov/learingData
cd 17/
find ./ -type f | sed 's!^.*/!!' | xargs -i cp {}  ~/face2/origin_image/16.{}
cd /mnt/mov/learingData
cd 18/
find ./ -type f | sed 's!^.*/!!' | xargs -i cp {}  ~/face2/origin_image/17.{}
cd /mnt/mov/learingData
cd 19/
find ./ -type f | sed 's!^.*/!!' | xargs -i cp {}  ~/face2/origin_image/18.{}
cd /mnt/mov/learingData

