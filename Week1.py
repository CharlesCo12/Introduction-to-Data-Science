import numpy as np
import matplotlib.pylab as plt
t=np.loadtxt("numeros_20.txt")
y=t[:,1]
x=t[:,0]
plt.scatter(t[:,0],t[:,1],color='blue',s=6)

def AproxFit2(x,y,M):
    #Creación de la Matriz de coeficientes
    def Coef(x,M):
        A=[]
        for i in range(len(x)):
            B=[]
            for j in range(M+1):
                B.append(x[i]**j)
            A.append(B)
        return A
    
    #Declaración de las Matrices para el producto matricial
    S=np.array(Coef(x,M))
    S1=np.linalg.pinv(S)
    c=np.dot(S1,y)
    
    return c
#Construcción de la Función
def f(x,c):
    suma=0
    for i in range(len(c)):
        y=c[i]*(x**i)
        suma+=y
    return suma
    
fig, axs = plt.subplots(2,2, figsize=(8, 8), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .4, wspace=.3)
axs=axs.ravel()

for i in range(4):
    xl=x[:i+2]
    yl=y[:i+2]
    c=AproxFit2(xl,yl,i+1)
    xm=np.linspace(np.min(xl),np.max(xl),100)
    axs[i].set_title("M=" + str(i+1))
    axs[i].plot(xm,f(xm,c))
    axs[i].scatter(xl,yl,color='black')
    axs[i].set_xlabel('x')
    axs[i].set_ylabel('y')
    
plt.savefig('solucion.png')