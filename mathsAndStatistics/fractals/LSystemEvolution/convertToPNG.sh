# requires imagemagick
for i in $(seq 0 29)
do
    convert $i.ps -background white -strip $i.png
    convert $i.png -background white -alpha remove -alpha off $i.png
done

# convert -delay 15 *.png -loop 0 -crop 600x675+60+25 animation.gif