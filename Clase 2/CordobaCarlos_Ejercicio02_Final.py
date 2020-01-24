import numpy as np
import matplotlib.pylab as plt
t=np.loadtxt("numeros_20.txt")
y=t[:,1]
x=t[:,0]

class Polyfit:
    degree = 0
    betas = []
    def __init__(self, degree):
        self.degree = degree
        self.betas = np.zeros(degree+1)
    
    def fit(self,x,y):
        M=self.degree
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
        self.betas+=np.array(c)
    
    def predict(self,x):
        suma=0
        y=0
        for i in range(len(self.betas)):
            y=self.betas[i]*(x**i)
            suma+=y
        return suma
    
    def score(self,x,y):
        N=len(x)
        a=self.predict(x)-y
        c=np.power(a,2)
        b=np.sum(c)
        return np.log(np.sqrt(b/N))

#Construcción de las gráficas para cada Potencia
fig, axs = plt.subplots(2,2, figsize=(10, 10), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .4, wspace=.3)
axs=axs.ravel()
M=[0,1,3,9]
xl=x[:10]
yl=y[:10]
xt=x[10:]
yt=y[10:]
for i in range(4):
    A=Polyfit(M[i])
    A.fit(xl,yl)
    c=A.betas
    xm=np.linspace(np.min(xl),np.max(xl),100)
    axs[i].set_title("M=" + str(M[i]))
    axs[i].plot(xm,A.predict(xm),label="M=" + str(M[i]),color='red')
    axs[i].scatter(xl,yl,facecolor='none',edgecolor='blue',s=40)
    axs[i].set_xlabel('$x$')
    axs[i].set_ylabel('$y$')
    axs[i].legend(loc=0)
plt.savefig('Gráficas.png')
plt.close()

#Construcción de la Gráfica del Erms
E_test=[]
E_training=[]
for i in range (10):
    D = Polyfit(i)
    D.fit(xl,yl)
    E_test.append(D.score(xt,yt))
    E_training.append(D.score(xl,yl))

q=np.arange(0,10)
plt.plot(q,E_test,label='Test',c='red', marker = 'o',markerfacecolor='none',markeredgecolor='red', ms=10)
plt.plot(q,E_training, label='Training',c='blue', marker = 'o',markerfacecolor='none',markeredgecolor='blue', ms=10)
plt.xlabel("M")
plt.ylabel("$log(E_{rms})$")
plt.legend()
loc=0
plt.savefig('Error.png')