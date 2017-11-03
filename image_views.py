# -*- coding: utf-8 -*-
"""
Created on Tue May  2 13:00:55 2017

@author: SzMike
"""

##############################################################
# PyPhoto 1.0: thumbnail image viewer with resizing and saves;
# supports multiple image directory thumb windows - the initial
# img dir is passed in as cmd arg, uses "images" default, or
# is selected via main window button; later directories are
# opened by pressing "D" in image view or thumbnail windows;
#
# viewer also scrolls popped-up images that are too large
# for the screen; still to do: (1) rearrange thumbnails when
# window resized, based on current window size; (2) resize
# images to fit current window size as an option? (3) avoid
# scrolls if image size is less than window max size: use
# Label if imgwide <= scrwide and imghigh <= scrhigh?
#
# New in 1.0: now does a form of (2) - image is resized to
# one of the display's dimensions if clicked, and zoomed in
# or out in 10% increments on key presses; generalize me;
# caveat: seems to lose quality, pixels after many resizes;
#
# the following scaler adapted from PIL's thumbnail code is
# similar to the screen height scaler here, but only shrinks:
# x, y = imgwide, imghigh
# if x > scrwide: y = max(y * scrwide / x, 1); x = scrwide
# if y > scrhigh: x = max(x * scrhigh / y, 1); y = scrhigh
##############################################################

import sys, math, os
import tkinter as tk
from tkinter.filedialog import SaveAs
from collections import OrderedDict

from PIL import Image                                  # PIL Image: also in Tkinter
from PIL.ImageTk import PhotoImage                # PIL photo widget replacement

# remember last dirs across all windows
#openDialog = Directory(title='Select Image Directory To Open')
appname = 'RSG 1.0: '

saveDialog = SaveAs(title='Save As (filename gives image type)')


class ScrolledCanvas(tk.Canvas):
    """
    a canvas in a container that automatically makes
    vertical and horizontal scroll bars for itself
    """
    def __init__(self, container):
        tk.Canvas.__init__(self, container)
        self.config(borderwidth=0)
        vbar = tk.Scrollbar(container)
        hbar = tk.Scrollbar(container, orient='horizontal')

        vbar.pack(side=tk.RIGHT,  fill=tk.Y)                 # pack canvas after bars
        hbar.pack(side=tk.BOTTOM, fill=tk.X)                 # so clipped first
        self.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)

        vbar.config(command=self.yview)                # call on scroll move
        hbar.config(command=self.xview)
        self.config(yscrollcommand=vbar.set)           # call on canvas move
        self.config(xscrollcommand=hbar.set)


class ViewOne(tk.Toplevel):
    """
    open a single image in a pop-up window when created;
    a class because photoimage obj must be saved, else
    erased if reclaimed; scroll if too big for display;
    on mouse clicks, resizes to window's height or width:
    stretches or shrinks; on I/O keypress, zooms in/out;
    both resizing schemes maintain original aspect ratio;
    code is factored to avoid redundancy here as possible;
    """
    def __init__(self, imgdir, imgfile, forcesize=( )):
        tk.Toplevel.__init__(self)
        helptxt = '(click L/R or press I/O to resize, S to save, D to open)'
        self.title(appname + imgfile + '  ' + helptxt)
        imgpath = os.path.join(imgdir, imgfile)
        imgpil  = Image.open(imgpath)
        self.canvas = ScrolledCanvas(self)
        self.drawImage(imgpil, forcesize)
        self.canvas.bind('<Button-1>', self.onSizeToDisplayHeight)
        self.canvas.bind('<Button-3>', self.onSizeToDisplayWidth)
        self.bind('<KeyPress-i>',      self.onZoomIn)
        self.bind('<KeyPress-o>',      self.onZoomOut)
        #self.bind('<KeyPress-s>',      self.onSaveImage)
        #self.bind('<KeyPress-d>',      onDirectoryOpen)
        self.focus( )

    def drawImage(self, imgpil, forcesize=( )):
        imgtk = PhotoImage(image=imgpil)                  # not file=imgpath
        scrwide, scrhigh = forcesize or self.maxsize( )   # wm screen size x,y
        imgwide  = imgtk.width( )                         # size in pixels
        imghigh  = imgtk.height( )                        # same as imgpil.size

        fullsize = (0, 0, imgwide, imghigh)              # scrollable
        viewwide = min(imgwide, scrwide)                 # viewable
        viewhigh = min(imghigh, scrhigh)

        canvas = self.canvas
        canvas.delete('all')                             # clear prior photo
        canvas.config(height=viewhigh, width=viewwide)   # viewable window size
        canvas.config(scrollregion=fullsize)             # scrollable area size
        canvas.create_image(0, 0, image=imgtk, anchor=tk.NW)

        if imgwide <= scrwide and imghigh <= scrhigh:    # too big for display?
            self.state('normal')                         # no: win size per img
        elif sys.platform[:3] == 'win':                  # do windows fullscreen
            self.state('zoomed')                         # others use geometry( )
        self.saveimage = imgpil
        self.savephoto = imgtk                           # keep reference on me
        print (scrwide, scrhigh), imgpil.size

    def sizeToDisplaySide(self, scaler):
        # resize to fill one side of the display
        imgpil = self.saveimage
        scrwide, scrhigh = self.maxsize( )                 # wm screen size x,y
        imgwide, imghigh = imgpil.size                     # img size in pixels
        newwide, newhigh = scaler(scrwide, scrhigh, imgwide, imghigh)
        if (newwide * newhigh < imgwide * imghigh):
            filter = Image.ANTIALIAS                      # shrink: antialias
        else:                                             # grow: bicub sharper
            filter = Image.BICUBIC
        imgnew  = imgpil.resize((newwide, newhigh), filter)
        self.drawImage(imgnew)

    def onSizeToDisplayHeight(self, event):
        def scaleHigh(scrwide, scrhigh, imgwide, imghigh):
            newhigh = scrhigh
            newwide = int(scrhigh * (float(imgwide) / imghigh))
            return (newwide, newhigh)
        self.sizeToDisplaySide(scaleHigh)

    def onSizeToDisplayWidth(self, event):
        def scaleWide(scrwide, scrhigh, imgwide, imghigh):
            newwide = scrwide
            newhigh = int(scrwide * (float(imghigh) / imgwide))
            return (newwide, newhigh)
        self.sizeToDisplaySide(scaleWide)

    def zoom(self, factor):
        # zoom in or out in increments
        imgpil = self.saveimage
        wide, high = imgpil.size
        if factor < 1.0:                     # antialias best if shrink
            filter = Image.ANTIALIAS         # also nearest, bilinear
        else:
            filter = Image.BICUBIC
        new = imgpil.resize((int(wide * factor), int(high * factor)), filter)
        self.drawImage(new)

    def onZoomIn(self, event, incr=.10):
        self.zoom(1.0 + incr)
    def onZoomOut(self, event, decr=.10):
        self.zoom(1.0 - decr)

    def onSaveImage(self, event):
        # save current image state to file
        filename = saveDialog.show( )
        if filename:
           self.saveimage.save(filename)


def viewThumbs(image_label, set_id, kind=tk.Toplevel, numcols=None, height=600, width=800):
    """
    make main or pop-up thumbnail buttons window;
    uses fixed-size buttons, scrollable canvas;
    sets scrollable (full) size, and places
    thumbs at abs x,y coordinates in canvas;
    no longer assumes all thumbs are same size:
    gets max of all (x,y), some may be smaller;
    """
    first_element=list(image_label.keys())[0]
    img_dir=os.path.dirname(first_element)
    view_list={}
    if type(list(image_label.values())[0])==OrderedDict:       
        meta=True
        for lists in image_label.values():
        # select bests
           first_file=list(lists.keys())[0]
           view_list[first_file]=list(lists.values())[0]
           os.path.basename(first_file).split('.')[0]
           thumbs = makeThumbs(list(lists.keys()),subdir=first_file.split('.')[0]) 
    else:
        meta=False
        view_list=image_label

    
    win = kind( )
    win.title(set_id)
    canvas = ScrolledCanvas(win)
    canvas.config(height=height, width=width)       # init viewable window size
                                                    # changes if user resizes
    
    # create thumbnails for the images
    thumbs = makeThumbs(list(view_list.keys()),subdir=set_id)                     # [(imgfile, imgobj)]
    numthumbs = len(thumbs)
    if not numcols:
        numcols = int(math.ceil(math.sqrt(numthumbs)))  # fixed or N x N
    numrows = int(math.ceil(numthumbs / float(numcols)))

    # thumb=(name, obj), thumb.size=(width, height)
    linksize = max([max(thumb[1].size) for thumb in thumbs])+15
    print(linksize)
    fullsize = (0, 0,                                   # upper left  X,Y
        (linksize * numcols), (linksize * numrows) )    # lower right X,Y
    canvas.config(scrollregion=fullsize)                # scrollable area size

    rowpos = 0
    savephotos = []
    while thumbs:
        thumbsrow, thumbs = thumbs[:numcols], thumbs[numcols:]
        colpos = 0
        for (imgfile, imgobj) in thumbsrow:
            photo   = PhotoImage(imgobj)
            link    = tk.Button(canvas, text=str(view_list[imgfile]),image=photo,compound='bottom')
            if meta:
                handler = (lambda savefile=imgfile: viewThumbs(image_label[savefile], os.path.basename(savefile).split('.')[0]))                
            else:
                handler = (lambda savefile=imgfile: ViewOne(img_dir, savefile))

            link.config(command=handler, width=linksize, height=linksize)
            link.pack(side=tk.LEFT, expand=tk.YES)
            canvas.create_window(colpos, rowpos, anchor=tk.NW,
                    window=link, width=linksize, height=linksize)
            colpos += linksize
            savephotos.append(photo)
        rowpos += linksize
    #win.bind('<KeyPress-d>', onDirectoryOpen)
    win.savephotos = savephotos
    return win


def makeThumbs(image_list, size=(150, 150), subdir='thumbs'):
    """
    get thumbnail images for all images in a directory;
    for each image, create and save a new thumb, or load
    and return an existing thumb; makes thumb dir if needed;
    returns list of (image filename, thumb image object);
    the caller can also run listdir on thumb dir to load;
    on bad file types we may get IOError, or other: overflow
    """
    imgdir=os.path.dirname(image_list[0])
    print(imgdir)
    thumbdir = os.path.join(imgdir, subdir)
    if not os.path.exists(thumbdir):
        os.mkdir(thumbdir)

    thumbs = []
    for imgfile in image_list:
        #print(imgfile)
        thumbpath = os.path.join(thumbdir, os.path.basename(imgfile))
        #print(thumbpath)
        if os.path.exists(thumbpath):
            thumbobj = Image.open(thumbpath)            # use already created
            thumbs.append((imgfile, thumbobj))
        else:
            #print 'making', thumbpath
            imgpath = os.path.join(imgdir, os.path.basename(imgfile))
            #print(thumbpath)
            try:
                imgobj = Image.open(imgpath)            # make new thumb
                imgobj.thumbnail(size, Image.ANTIALIAS) # best downsize filter
                imgobj.save(thumbpath)                  # type via ext or passed
                thumbs.append((imgfile, imgobj))
            except:                                     # not always IOError
                print('Skipping: '+imgpath)
    return thumbs