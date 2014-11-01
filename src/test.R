x <- c(10.4, 5.6, 3.1, 6.4, 21.7)
y <- c(36, 74. ,3,57, 97)
xy <- outer(x,y,"*")

ans <- diffuse(xy)
