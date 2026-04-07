""" Based on the algorithm in "EFFECTIVE METHODS FOR CONSTRUCTING
EXTREME QUANTUM OBSERVABLES", link:https://arxiv.org/pdf/1809.09935"""

import numpy as np

#find dimension
d=4 #dimension of the hilbert place
r=int(input("No of outcomes:"))

#get rank information
m=np.zeros(r,dtype="int")
for i in range(r):
    m_i=int(input("Rank of POVM element {}:".format(i+1)))
    m[i]=m_i

#construct the square
n_v=np.reshape(np.identity(d),(d,d)) #generates orthonormal set of form {e_0,e_1...}
Bj=np.ndarray((r,2),dtype="int")

Bj[0]=[0,2]
Bj[1]=[2,0]
Bj[2]=[1,3]
Bj[3]=[3,1]
Bj[4]=[1,2]
Bj[5]=[2,1]
Bj[6]=[3,2]
Bj[7]=[2,3]
Bj[8]=[0,3]
Bj[9]=[3,0]


#generate |h_{jk}>s


#generate R
def h(j,k):
    r,s=Bj[j-1][0]+k-1,Bj[j-1][1]+k-1
    #print("r,s:",r,s)
    #print("r,s:",r,s)
    if r==s:
        return (n_v[r])
    elif r>s:
        return (n_v[r]+n_v[s])
    else:
        return (n_v[r]-1j*n_v[s])

def R_j_gen(j):
    R_j_o=np.zeros((d,d),dtype='complex') #check notation of matrix properly
    for k in range(m[j-1]):
        k=k+1 #k also starts from 1
        h_jk=h(j,k)
        R_j_k= np.outer(h_jk,h_jk.conj())
        R_j_o+=R_j_k
    return R_j_o

R2_inv=np.zeros((d,d),dtype='complex')   
for cor in range(r):
    R_j=R_j_gen(cor+1) #numbering of POVMs start from 1
    R2_inv+=R_j 
eigval,eigvec=np.linalg.eig(R2_inv)
assert (eigval>=0).all()
R=eigvec*(1/np.sqrt(eigval)) @ np.linalg.inv(eigvec)

#generate Mjs
M=np.zeros((r,d,d),dtype='complex')
for j in range(r):
    j=j+1 #j starts from 1
    M_j=np.zeros((d,d),dtype='complex')
    m_js=np.zeros((d,d),dtype='complex')
    for k in range(m[j-1]):
        k=k+1 #k starts from 1
        h_jk=h(j,k)
        R_j_k= np.outer(h_jk,h_jk.conj())
        m_js+=R_j_k 
    M_j+=R @ m_js @ R
    M[j-1]=M_j

print("M[i] prints the (i+1)th POVM")
        
#calculating entropy
I2=np.identity(2)
rho=(1/4)*np.kron(I2,I2)
h,j=0,0
for i in range(10):
    p_i=np.real(np.trace(M[i]@rho))
    print("p({})=".format(i+1),p_i)
    j+=np.trace(M[i]@rho)
    s=-1*p_i*np.log2(p_i)
    h+=s
print("entropy=",h)


  
