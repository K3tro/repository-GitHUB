#=============================================================================
import GOES
from tkinter import*
from tkinter import messagebox
from tkinter import filedialog
from tkinter import ttk
from tkinter.ttk import *
from datetime import datetime, date, time, timedelta
import cftime####
import cftime._strptime####
import babel.numbers####
import os                            # Miscellaneous operating system interfaces
import matplotlib.pyplot as plt            # Plotting library
import cartopy, cartopy.crs as ccrs        # Plot maps
import numpy as np                         # Scientific computing with Python
from glob import glob as gb
from netCDF4 import Dataset
import colorsys
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
import threading
from tkcalendar import Calendar, DateEntry
from idlelib.tooltip import Hovertip
from PIL import ImageTk, Image

#============================FUNCIONES_INTERNAS=================================================
def validate_float(action, index, value_if_allowed,
    prior_value, text, validation_type, trigger_type, widget_name):
    # action=1 -> insert
    if(action=='1'):
        if text in '+-0123456789.':
            try:
                float(value_if_allowed)
                return True
            except ValueError:
                return False
        else:
            return False
    else:
        return True
def cambio(buton,val):
    if buton['style']=='on.TButton':
        buton.config(style='off.TButton');bands.remove(val)
    elif buton['style']=='off.TButton':
        buton.config(style='on.TButton');bands.append(val)
def loadCPT(path):

    path=r'paletas/'+path
    try:
        f = open(path)
    except:
        print ("File ", path, "not found")
        return None

    lines = f.readlines()
    f.close()
    x = np.array([])
    r = np.array([])
    g = np.array([])
    b = np.array([])
    colorModel = 'RGB'
    for l in lines:
        ls = l.split()
        if l[0] == '#':
            if ls[-1] == 'HSV':
                colorModel = 'HSV'
                continue
            else:
                continue
        if ls[0] == 'B' or ls[0] == 'F' or ls[0] == 'N':
            pass
        else:
            x=np.append(x,float(ls[0]))
            r=np.append(r,float(ls[1]))
            g=np.append(g,float(ls[2]))
            b=np.append(b,float(ls[3]))
            xtemp = float(ls[4])
            rtemp = float(ls[5])
            gtemp = float(ls[6])
            btemp = float(ls[7])

        x=np.append(x,xtemp)
        r=np.append(r,rtemp)
        g=np.append(g,gtemp)
        b=np.append(b,btemp)
    if colorModel == 'HSV':
        for i in range(r.shape[0]):
            rr, gg, bb = colorsys.hsv_to_rgb(r[i]/360.,g[i],b[i])
        r[i] = rr ; g[i] = gg ; b[i] = bb
    if colorModel == 'RGB':
        r = r/255.0
        g = g/255.0
        b = b/255.0
    xNorm = (x - x[0])/(x[-1] - x[0])
    red   = []
    blue  = []
    green = []
    for i in range(len(x)):
        red.append([xNorm[i],r[i],r[i]])
        green.append([xNorm[i],g[i],g[i]])
        blue.append([xNorm[i],b[i],b[i]])
    colorDict = {'red': red, 'green': green, 'blue': blue}
    return colorDict

def graficar(product,i,eextent=[-110,-30,-60,15],save=False):
    def f(band,Fecha):
        return sorted(gb(f'GOES/*C{str(band).zfill(2)}**{Fecha}*.nc'))[0]
        
    rgb=False;band=False
    invR=False;invG=False;invB=False
    #Combinaciones RGB
    if product=='air mass':
        base=Dataset(f(8,i))
        GOES_8=base.variables['CMI'][:]
        GOES_10=Dataset(f(10,i)).variables['CMI'][:]
        GOES_12=Dataset(f(12,i)).variables['CMI'][:]
        GOES_13=Dataset(f(13,i)).variables['CMI'][:]
        R=GOES_8.data-GOES_10.data;G=GOES_12.data-GOES_13.data;B=GOES_8.data-273;Γ=[1,1,1]
        rangos=[[-26.2,0.6],[-43.2,6.7],[-64.65,-29.25]];invB=True;rgb=True;titulo='Air Mass'
    elif product=='day land cloud':
        base=Dataset(f(5,i))
        GOES_5=base.variables['CMI'][:]
        GOES_3=Dataset(f(3,i)).variables['CMI'][:]
        GOES_2=Dataset(f(2,i)).variables['CMI'][:]
        R=GOES_5.data[:][::4,::4];G=GOES_3.data[::4,::4];B=GOES_2.data[::8,::8]
        Γ=[1,1,1];rangos=[[0,0.975],[0,1.086],[0,1]];rgb=True;titulo='Day Land Cloud'
    elif product=='so2':
        base=Dataset(f(9,i))
        GOES_9=base.variables['CMI'][:]
        GOES_10=Dataset(f(10,i)).variables['CMI'][:]
        GOES_11=Dataset(f(11,i)).variables['CMI'][:]
        GOES_13=Dataset(f(13,i)).variables['CMI'][:]
        R=GOES_9.data-GOES_10.data;G=GOES_13.data-GOES_11.data;B=GOES_13.data-273;Γ=[1,1,1]
        rangos=[[-4.0,2.0],[-4.0,5.0],[-30.1,29.8]];rgb=True;titulo='SO2'
    elif product=='NtMicro':
        base=Dataset(f(7,i))
        GOES_7=base.variables['CMI'][:]
        GOES_15=Dataset(f(15,i)).variables['CMI'][:]
        GOES_13=Dataset(f(13,i)).variables['CMI'][:]
        R=GOES_15.data-GOES_13.data;G=GOES_13.data-GOES_7.data;B=GOES_13.data-273;Γ=[1,1,1]
        rangos=[[-6.7,2.6],[-3.1,5.2],[-29.6,19.5]];rgb=True;titulo='Nighttime Microphysics'
    elif product=='NTrueC':
        base=Dataset(f(1,i))
        GOES_1=base.variables['CMI'][:]
        GOES_2=Dataset(f(2,i)).variables['CMI'][:]
        GOES_3=Dataset(f(3,i)).variables['CMI'][:]
        R=GOES_2.data[:][::4,::4];G=0.45*(GOES_2[:][::4,::4])+0.1*(GOES_3[:][::2,::2])+0.45*(GOES_1[:][::2,::2]);B=GOES_1.data[:][::2,::2]
        Γ=[1,1,1];rangos=[[0,0.975],[0,1.086],[0,1]];rgb=True;titulo='CIMSS Natural True Color'
    elif product=='DCloudPD':
        base=Dataset(f(13,i))
        GOES_13=base.variables['CMI'][:]
        GOES_2=Dataset(f(2,i)).variables['CMI'][:]
        GOES_5=Dataset(f(5,i)).variables['CMI'][:]
        R=GOES_13.data[:][::2,::2]-273;G=GOES_2.data[:][::8,::8];B=GOES_5.data[:][::4,::4]########################
        Γ=[1,1,1];rangos=[[-53.5,-7.5],[0,0.78],[0.01,0.59]];rgb=True;titulo='Day Cloud Phase Distinction';invR=True
    #Diferencia de Bandas
    elif product=='split ozone':
        base=Dataset(f(12,i))
        GOES_12=base.variables['CMI'][:]
        GOES_13=Dataset(f(13,i)).variables['CMI'][:]
        RGB=GOES_12-GOES_13;band=True
        cpt=WMBSGGYRW
        cpt_convert=LinearSegmentedColormap('cpt',cpt);lim=[-45,5];titulo='Split Ozone'
    elif product=='split water':
        base=Dataset(f(8,i))
        GOES_8=base.variables['CMI'][:]
        GOES_10=Dataset(f(10,i)).variables['CMI'][:]
        RGB=GOES_8-GOES_10;band=True
        cpt=WMBSGGYRW
        cpt_convert=LinearSegmentedColormap('cpt',cpt);lim=[-35,5];titulo='Split Water Vapor Difference'
    #Bandas Simples
    elif product=='b7':
        base=Dataset(f(product[1:],i));RGB=base.variables['CMI'][:]-273;band=True
        cpt=SVGAIR2_TEMP
        cpt_convert=LinearSegmentedColormap('cpt',cpt);lim=[-112.15,56.85];titulo='Shortwave Window'
    elif product=='b2':
        base=Dataset(f(product[1:],i));RGB=base.variables['CMI'][:][::8,::8]*100;band=True
        cpt=BKANDWY
        cpt_convert=LinearSegmentedColormap('cpt',cpt);lim=[0,100];titulo="“Veggie”"
    elif product=='b3':
        base=Dataset(f(product[1:],i));RGB=base.variables['CMI'][:][::4,::4]*100;band=True
        cpt=BKANDWY
        cpt_convert=LinearSegmentedColormap('cpt',cpt);lim=[0,100];titulo="Red"
    elif product=='b9' or product=='b8' or product=='b10':
        base=Dataset(f(product[1:],i));RGB=base.variables['CMI'][:]-273;band=True
        cpt=SVGAWVX_TEMP
        cpt_convert=LinearSegmentedColormap('cpt',cpt);lim=[-112.15,56.85]
        titulo=['Upper-level tropospheric water vapor','Mid-level tropospheric water vapor','Lower-level tropospheric water vapor'][int(product[1:])-8]
    elif product=='b13' or product=='b14':
        base=Dataset(f(product[1:],i));RGB=base.variables['CMI'][:]-273;band=True
        cpt=IR4AVHRR6#IR4AVHRR6
        cpt_convert=LinearSegmentedColormap('cpt',cpt);lim=[-100,100];titulo=["'Clean' IR Longwave Window","IR Longwave Window"][int(product[1:])-13]
        
    if rgb:
        R=np.clip(R,rangos[0][0],rangos[0][1])
        G=np.clip(G,rangos[1][0],rangos[1][1])
        B=np.clip(B,rangos[2][0],rangos[2][1])
        def norm(color,Γ,r):
            return (((color-r[0])/(r[1]-r[0]))**(1/Γ))
        R=norm(R,Γ[0],rangos[0])
        G=norm(G,Γ[1],rangos[1])
        B=norm(B,Γ[2],rangos[2])
        if invR:R=1-R
        elif invG:G=1-G
        elif invB:B=1-B
        RGB=np.stack([R,G,B],axis=2)
    lcolor='black';d1=abs(eextent[0]-eextent[1]);d2=abs(eextent[2]-eextent[3])
    fig = plt.figure(figsize=[25*d1/d2, 25], dpi=100)
    ax = fig.add_subplot(1,1,1)
    plt.axis('off')
    
    lon_sat=base.variables['goes_imager_projection'].longitude_of_projection_origin
    lev_sat=base.variables['goes_imager_projection'].perspective_point_height
    ax=plt.axes(projection=ccrs.PlateCarree(central_longitude=lon_sat))

    xmin=base.variables['x'][:].min()*lev_sat
    xmax=base.variables['x'][:].max()*lev_sat
    ymin=base.variables['y'][:].min()*lev_sat
    ymax=base.variables['y'][:].max()*lev_sat
    img_extent=np.array([xmin,xmax,ymin,ymax])

    ax.set_extent(eextent,crs=ccrs.PlateCarree())
    ax.stock_img()
    ax.add_feature(cartopy.feature.BORDERS,edgecolor=lcolor,linewidth=1)
    ax.coastlines(resolution='110m',color=lcolor,linewidth=2)

    grid_size=round(((d1+d2)/2)/4)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), color='dimgray', alpha=1.0, linestyle='--',
                      linewidth=0.25, xlocs=np.arange(-180, 180, grid_size), 
                      ylocs=np.arange(-90, 90, grid_size), draw_labels=True)
    gl.right_labels=False
    gl.left_labels=True
    gl.bottom_labels=False
    gl.top_labels =True
    gl.xpadding = -5 #distancia desde el borde para los numeros
    gl.ypadding = -5
    gl.xlabel_style = {'size': 25,'color': 'white'}#coloreamos los numeros
    gl.ylabel_style = {'size': 25,'color': 'white'}
    
    p = patches.Rectangle((0, 0), 1, 0.02,fill=True, transform=ax.transAxes, clip_on=False,color='dimgray')#xy, width, height
    Date=base.time_coverage_start[:-12]+" "+base.time_coverage_start[11:-6]
    ax.text(0, 0.002,transform=ax.transAxes,s=Date + ' UTC G-16 IMG '+titulo,fontsize=25,color='white',)
    
    if band: 
        img=ax.imshow(RGB,origin='upper',vmin=lim[0],vmax=lim[1],extent=img_extent,cmap=cpt_convert,transform=ccrs.Geostationary(central_longitude=lon_sat,satellite_height=lev_sat))
        cax = ax.inset_axes([0,  0.02, 1, 0.015])#[x0, y0, width, height]
        cb=fig.colorbar(img,orientation="horizontal",cax=cax)
        cb.outline.set_visible(False)
        cb.ax.xaxis.set_tick_params(pad=-22.5)
        cb.ax.tick_params(axis='x', colors='black', labelsize=22)
    elif rgb:
        img=ax.imshow(RGB,origin='upper',extent=img_extent,transform=ccrs.Geostationary(central_longitude=lon_sat,satellite_height=lev_sat))
    ax.add_patch(p)
    if save:
        Date=Date.replace(' ','--').replace(':','-')
        plt.savefig(f'IMGS/{product} '+Date+ '.png')
    #plt.show()
    plt.close()
     
#===========================FUNCION_DE_BOTONES==================================================

def AbrirFichero():
    Fichero=filedialog.askdirectory(title='Abrir')
    os.chdir(Fichero)
    os.getcwd()
    fichero.set(Fichero)
def CrearCarpetas():
    if fichero.get()!='':
        dir = "GOES"; os.makedirs(dir, exist_ok=True)
        dir = "IMGS"; os.makedirs(dir, exist_ok=True)
        dir = "GIFS"; os.makedirs(dir, exist_ok=True)
    else:messagebox.showwarning("Aviso:","Por favor, eliga una ruta de guardado primero")

def DESCARGATELA():
    if fichero.get()=='':return messagebox.showwarning("Aviso:","Por favor, eliga una ruta de guardado primero")
    CrearCarpetas()
    t1=datetime.strptime(cal1.get(),'%m/%d/%y')+timedelta(hours=int(h1.get()))
    t2=datetime.strptime(cal2.get(),'%m/%d/%y')+timedelta(hours=int(h2.get()))
    try:
        intervalo=timedelta(hours=int(interH.get()),minutes=int(interM.get()))
    except:
        return messagebox.showwarning("Aviso:",'Ingrese un intervalo de tiempo valido!!')
    fatidico_dia=datetime(2019,4,3)
    null=timedelta(0)
    if t2<fatidico_dia and intervalo%timedelta(minutes=15)!=null:intervalo=(intervalo//timedelta(minutes=15)+1)*timedelta(minutes=15)
    elif t2>fatidico_dia:
        if t1>fatidico_dia and intervalo%timedelta(minutes=10)!=null:
            intervalo=(intervalo//timedelta(minutes=10)+1)*timedelta(minutes=10)
        elif t1<fatidico_dia and intervalo%timedelta(minutes=30)!=null:
            intervalo=(intervalo//timedelta(minutes=30)+1)*timedelta(minutes=30)  
    #3 Abril 2019
    canales=[str(i).zfill(2) for i in bands]
    Tiempos=[t1+intervalo*i for i in np.arange(0,(t2-t1)/intervalo)]
    
    if Tiempos==[]:return messagebox.showwarning("Aviso:","La fecha final debe ser posterior a la fecha inicial!!")
    if canales==[]:return messagebox.showwarning("Aviso:",'Se debe escoger al menos una banda!!')
    carg=Tk()
    carg.geometry("300x50") 
    carg.title("Descargando Datos ....")
    def xd():
        pb = ttk.Progressbar(carg,orient='horizontal',mode='determinate',length=280)
        pb.grid(column=0, row=0, columnspan=2, padx=10, pady=20)
        confirmo.destroy()
        niego.destroy()
        for i in Tiempos:
            ini=i
            end=ini+timedelta(minutes=9)
            inicio=datetime.strftime(ini,'%Y%m%d-%H%M00')
            fin=datetime.strftime(end,'%Y%m%d-%H%M00')
            GOES.download('goes16', 'ABI-L2-CMIPF',DateTimeIni = inicio,DateTimeFin=fin,channel =canales,rename_fmt = '%Y-%m-%d--%H-%M-%S',show_download_progress=True, path_out='GOES/')
            pb['value'] +=100/len(Tiempos)
            carg.update()
        carg.destroy()
    confirmo=Button(carg,text='Aceptar',command=lambda: threading.Thread(target=xd).start());confirmo.place(height=30, width=90,x=40, y=10)
    niego=Button(carg,text='Cancelar',command=lambda: carg.destroy());niego.place(height=30, width=90,x=170, y=10)
    carg.mainloop()
    return print('fin')
    
    
def GRAFICATELA():
    if fichero.get()=='':return messagebox.showwarning("Aviso:","Por favor, eliga una ruta de guardado primero")
    
    t1=datetime.strptime(cal1.get(),'%m/%d/%y')+timedelta(hours=int(h1.get()))
    t2=datetime.strptime(cal2.get(),'%m/%d/%y')+timedelta(hours=int(h2.get()))
    try:
        intervalo=timedelta(hours=int(interH.get()),minutes=int(interM.get()))
    except:
        return messagebox.showwarning("Aviso:",'Ingrese un intervalo de tiempo valido!!')
    fatidico_dia=datetime(2019,4,3)
    null=timedelta(0)
    if t2<fatidico_dia and intervalo%timedelta(minutes=15)!=null:intervalo=(intervalo//timedelta(minutes=15)+1)*timedelta(minutes=15)
    elif t2>fatidico_dia:
        if t1>fatidico_dia and intervalo%timedelta(minutes=10)!=null:
            intervalo=(intervalo//timedelta(minutes=10)+1)*timedelta(minutes=10)
        elif t1<fatidico_dia and intervalo%timedelta(minutes=30)!=null:
            intervalo=(intervalo//timedelta(minutes=30)+1)*timedelta(minutes=30) 
    
    Tiempos=[t1+intervalo*i for i in np.arange(0,(t2-t1)/intervalo)]
    
    if Tiempos==[]:return messagebox.showwarning("Aviso:","La fecha final debe ser posterior a la fecha inicial!!")
    extent=[float(e.get()),float(o.get()),float(s.get()),float(n.get())]
    products=[i.get() for i in Plist if i.get()!='']
    if products==[]:return messagebox.showwarning("Aviso:",'Se debe escoger al menos un producto!!')
    Time=[j.strftime('%Y-%m-%d--%H-%M') for j in Tiempos]
    total=len(products)*len(Time)
    carg=Tk()
    #carg.iconphoto(False, sat_ico)
    carg.geometry("300x50") 
    carg.title("Graficando Imagenes ....")
    def xd():
        pb = ttk.Progressbar(carg,orient='horizontal',mode='determinate',length=280)
        pb.grid(column=0, row=0, columnspan=2, padx=10, pady=20)
        confirmo.destroy()
        niego.destroy()
        for p in products:
            for t in Time:
                w=graficar(p,t,extent,save=True)
                if w:
                    return messagebox.showwarning("Aviso:",'No se han descargado las bandas correspondientes todavia!!'),carg.destroy()
                
                pb['value'] +=100/total
                carg.update()
        carg.destroy()
    confirmo=Button(carg,text='Aceptar',command=lambda: threading.Thread(target=xd).start());confirmo.place(height=30, width=90,x=40, y=10)
    niego=Button(carg,text='Cancelar',command=lambda: carg.destroy());niego.place(height=30, width=90,x=170, y=10)
    carg.mainloop()
    return print('fin')
#===========================RAIZ==================================================
raiz= Tk() ;vcmd = (raiz.register(validate_float),'%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')
raiz.title("Imagenes Satelitales Goes-16")
miFrame=Frame(raiz);miFrame.grid(row=0,column=0,columnspan=2)
miFrame2=Frame(raiz);miFrame2.grid(row=1,column=0,padx=20)
miFrame3=Frame(raiz);miFrame3.grid(row=2,column=0,pady=30,columnspan=2)
miFrame4=Frame(raiz);miFrame4.grid(row=1,column=1,padx=20)
miFrame5=Frame(raiz);miFrame5.grid(row=0,column=2,padx=20)
miFrame6=Frame(raiz);miFrame6.grid(row=1,column=2,padx=20)
miFrame7=Frame(raiz);miFrame7.grid(row=2,column=2,padx=20)

ra = PhotoImage(file = "imagenes/sat.png")
#raiz.iconbitmap("imagenes/sat.ico")
raiz.iconphoto(False, ra)
#raiz.tk.call('wm', 'iconphoto', raiz._w,PhotoImage(file='imagenes/sat.png'))
#%%variables pre fijadas
fichero=StringVar()
intervaloH=StringVar()
intervaloM=StringVar()

IR4AVHRR6=loadCPT('IR4AVHRR6.cpt')
SVGAWVX_TEMP=loadCPT('SVGAWVX_TEMP.cpt')
SVGAIR2_TEMP=loadCPT('SVGAIR2_TEMP.cpt')
WMBSGGYRW=loadCPT('WMBSGGYRW.cpt')
BKANDWY=loadCPT('BKANDWY.cpt')
#=============miFrame(fecha intervalos ruta y boton de creacion de botones)==================================
yest=datetime.today()-timedelta(1)
fi=Label(miFrame,text="Fecha inicial:");fi.grid(row=0,column=0,sticky='w',padx=2,pady=2)
cal1=DateEntry(miFrame,selectmode='day',day=int(yest.strftime('%d')),year=int(yest.strftime('%Y')),month=int(yest.strftime('%m')));cal1.grid(row=0,column=1,padx=2,sticky='w')
h1 = ttk.Combobox(miFrame,state="readonly", width=3,values=np.arange(0,25).tolist());h1.grid(row=0,column=2)

ff=Label(miFrame,text="Fecha Final:");ff.grid(row=0,column=3,sticky='w',padx=2,pady=2)
cal2=DateEntry(miFrame,selectmode='day');cal2.grid(row=0,column=4,padx=2,sticky='w')
h2 = ttk.Combobox(miFrame,state="readonly", width=3,values=np.arange(0,25).tolist());h2.grid(row=0,column=5)

intxt=Label(miFrame,text="Intervalos de tiempo(h):");intxt.grid(row=0,column=6,sticky='w',padx=2,pady=2)
interH=Entry(miFrame,text=intervaloH,validate='key',validatecommand = vcmd);interH.config(width=2);interH.grid(row=0,column=7)
intxtt=Label(miFrame,text=":");intxtt.grid(row=0,column=8,sticky='w',padx=2,pady=2)
interM=Entry(miFrame,text=intervaloM,validate='key',validatecommand = vcmd);interM.config(width=2);interM.grid(row=0,column=9)

intervaloH.set('01')
intervaloM.set('00')
carpetImg = PhotoImage(file="imagenes/carpeta.png").subsample(20)
ruta=Entry(miFrame5,textvariable=fichero,justify=LEFT,state='readonly');ruta.config(width=50);ruta.grid(row=0,column=10,padx=2,sticky='E')
Setcarpet=Button(miFrame5,image=carpetImg,command=AbrirFichero);Setcarpet.grid(row=0,column=11,sticky='W')

Hovertip(Setcarpet, text="Buscar un Directorio de Trabajo", hover_delay=10)
#Hovertip(MadeCarpets, text="Crear Carpetas en el Directorio Seleccionado", hover_delay=10)

h1.set('0');h2.set('0')
#=============miFrame2_(zona de estudio)===================================================================

n=StringVar();n.set('0')#90
s=StringVar();s.set('-50')#-90
o=StringVar();o.set('-110')#180
e=StringVar();e.set('-30')#-180

ntext=Label(miFrame2,text="Norte");ntext.grid(row=0,column=1,sticky='w',padx=2)
stext=Label(miFrame2,text="Sur");stext.grid(row=4,column=1,sticky='w',padx=2)
etext=Label(miFrame2,text="Este");etext.grid(row=2,column=2,sticky='w',padx=2)
otext=Label(miFrame2,text="Oeste");otext.grid(row=2,column=0,sticky='w',padx=2)

N=Entry(miFrame2,textvariable=n,justify=CENTER, width=10);N.grid(row=1,column=1,sticky='e')
S=Entry(miFrame2,textvariable=s,justify=CENTER, width=10);S.grid(row=5,column=1,sticky='e')
E=Entry(miFrame2,textvariable=e,justify=CENTER, width=10);E.grid(row=3,column=2,sticky='e')
O=Entry(miFrame2,textvariable=o,justify=CENTER, width=10);O.grid(row=3,column=0,sticky='e')

#=============miFrame4_Seleccionar Bandas a descargar==================================

sto = Style();sto.configure('on.TButton', font= ('Arial', 10, 'underline'),foreground='Blue',background='Blue')
sta= Style();sta.configure('off.TButton', font= ('Arial', 10),foreground='Black')
bands=[];sep=1
bb1 = Button(miFrame4, text='Banda 1' ,style='off.TButton',command=lambda:cambio(bb1, 1 ),takefocus=False);bb1 .grid(row=0, column=0, padx=sep)
bb2 = Button(miFrame4, text='Banda 2' ,style='off.TButton',command=lambda:cambio(bb2, 2 ),takefocus=False);bb2 .grid(row=0, column=1, padx=sep)
bb3 = Button(miFrame4, text='Banda 3' ,style='off.TButton',command=lambda:cambio(bb3, 3 ),takefocus=False);bb3 .grid(row=0, column=2, padx=sep)
bb4 = Button(miFrame4, text='Banda 4' ,style='off.TButton',command=lambda:cambio(bb4, 4 ),takefocus=False);bb4 .grid(row=0, column=3, padx=sep)
bb5 = Button(miFrame4, text='Banda 5' ,style='off.TButton',command=lambda:cambio(bb5, 5 ),takefocus=False);bb5 .grid(row=1, column=0, padx=sep)
bb6 = Button(miFrame4, text='Banda 6' ,style='off.TButton',command=lambda:cambio(bb6, 6 ),takefocus=False);bb6 .grid(row=1, column=1, padx=sep)
bb7 = Button(miFrame4, text='Banda 7' ,style='off.TButton',command=lambda:cambio(bb7, 7 ),takefocus=False);bb7 .grid(row=1, column=2, padx=sep)
bb8 = Button(miFrame4, text='Banda 8' ,style='off.TButton',command=lambda:cambio(bb8, 8 ),takefocus=False);bb8 .grid(row=1, column=3, padx=sep)
bb9 = Button(miFrame4, text='Banda 9' ,style='off.TButton',command=lambda:cambio(bb9, 9 ),takefocus=False);bb9 .grid(row=2, column=0, padx=sep)
bb10= Button(miFrame4, text='Banda 10',style='off.TButton',command=lambda:cambio(bb10,10),takefocus=False);bb10.grid(row=2, column=1, padx=sep)
bb11= Button(miFrame4, text='Banda 11',style='off.TButton',command=lambda:cambio(bb11,11),takefocus=False);bb11.grid(row=2, column=2, padx=sep)
bb12= Button(miFrame4, text='Banda 12',style='off.TButton',command=lambda:cambio(bb12,12),takefocus=False);bb12.grid(row=2, column=3, padx=sep)
bb13= Button(miFrame4, text='Banda 13',style='off.TButton',command=lambda:cambio(bb13,13),takefocus=False);bb13.grid(row=3, column=0, padx=sep)
bb14= Button(miFrame4, text='Banda 14',style='off.TButton',command=lambda:cambio(bb14,14),takefocus=False);bb14.grid(row=3, column=1, padx=sep)
bb15= Button(miFrame4, text='Banda 15',style='off.TButton',command=lambda:cambio(bb15,15),takefocus=False);bb15.grid(row=3, column=2, padx=sep)
bb16= Button(miFrame4, text='Banda 16',style='off.TButton',command=lambda:cambio(bb16,16),takefocus=False);bb16.grid(row=3, column=3, padx=sep)
#=============miFrame3_Seleccionar producto==================================
p1=StringVar()
p2=StringVar()
p3=StringVar()
p4=StringVar()
p5=StringVar()
p6=StringVar()
p7=StringVar()
p8=StringVar()
p9=StringVar()
p10=StringVar()
p11=StringVar()
p12=StringVar()
p13=StringVar()
p14=StringVar()
p15=StringVar()
p16=StringVar()
Plist=[p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16]

prgb=Label(miFrame3,text="Productos RGB");prgb.grid(row=0,column=0,padx=2)
psplt=Label(miFrame3,text="Diferencia de Canales");psplt.grid(row=0,column=1,padx=2)
pbnd=Label(miFrame3,text="Canales Individuales");pbnd.grid(row=0,column=2,padx=2)

o1=Checkbutton(miFrame3,text='Air Mass\n(8 10 12 13)',variable=p1,onvalue="air mass",offvalue="").grid(row=1,column=0,sticky='w')
o2=Checkbutton(miFrame3,text='Day Land Cloud\n(2 3 5)',variable=p2,onvalue="day Land Cloud",offvalue="").grid(row=2,column=0,sticky='w')
o3=Checkbutton(miFrame3,text='So2\n(9 10 11 13)',variable=p3,onvalue="so2",offvalue="").grid(row=3,column=0,sticky='w')
o4=Checkbutton(miFrame3,text='Nighttime\nMicrophysics (7 15 13)',variable=p4,onvalue="NtMicro",offvalue="").grid(row=4,column=0,sticky='w')
o5=Checkbutton(miFrame3,text='CIMSS Natural True Color\n(1 2 3)',variable=p5,onvalue="NTrueC",offvalue="").grid(row=5,column=0,sticky='w')
o14=Checkbutton(miFrame3,text='Day Cloud Phase Distinction\n(2 5 13)',variable=p14,onvalue="DCloudPD",offvalue="").grid(row=6,column=0,sticky='w')

o6=Checkbutton(miFrame3,text='Split Ozone\n(12 13)',variable=p6,onvalue="split ozone",offvalue="").grid(row=1,column=1,sticky='w')
o7=Checkbutton(miFrame3,text='Split Water Vapor\nDifference (8 10)',variable=p7,onvalue="split water",offvalue="").grid(row=2,column=1,sticky='w')

o15 =Checkbutton(miFrame3,text="Red\nBanda 2 (0.64 μm)",variable=p15,onvalue="b2",offvalue="").grid(row=1,column=2,sticky='w')
o15 =Checkbutton(miFrame3,text="“Veggie”\nBanda 3 (0.86 μm)",variable=p16,onvalue="b3",offvalue="").grid(row=2,column=2,sticky='w')
o8  =Checkbutton(miFrame3,text='Shortwave Window Banda 7 (3.9 μm)',variable=p8,onvalue="b7",offvalue="").grid(row=3,column=2,sticky='w')
o9  =Checkbutton(miFrame3,text='Upper-level tropospheric water vapor\nBanda 8 (6.2 μm)',variable=p9,onvalue="b8",offvalue="").grid(row=4,column=2,sticky='w')
o10 =Checkbutton(miFrame3,text='Mid-level tropospheric water vapor\nBanda 9 (6.9 μm)',variable=p10,onvalue="b9",offvalue="").grid(row=5,column=2,sticky='w')
o11 =Checkbutton(miFrame3,text='Lower-level tropospheric water vapor\nBanda 10 (7.3 μm)',variable=p11,onvalue="b10",offvalue="").grid(row=6,column=2,sticky='w')
o12 =Checkbutton(miFrame3,text="'Clean' IR Longwave Window\nBanda 13 (10.3 μm)",variable=p12,onvalue="b13",offvalue="").grid(row=7,column=2,sticky='w')
o13 =Checkbutton(miFrame3,text="IR Longwave Window\nBanda 14 (11.2 μm)",variable=p13,onvalue="b14",offvalue="").grid(row=8,column=2,sticky='w')

#%%Variables Obtenidas
downImg = PhotoImage(file="imagenes/download.png").subsample(12)
engrImg = PhotoImage(file="imagenes/engranes.png").subsample(10)

Descarga=Button(miFrame6,image=downImg,command=lambda: threading.Thread(target=DESCARGATELA).start());Descarga.grid(row=3,column=5,sticky='e')
Creacion=Button(miFrame7,image=engrImg,command=lambda: threading.Thread(target=GRAFICATELA).start());Creacion.grid(row=3,column=6,sticky='e')

Hovertip(Descarga, text="Descargar", hover_delay=10)
Hovertip(Creacion, text="Crear Imagenes", hover_delay=10)
#%%Show
raiz.mainloop()