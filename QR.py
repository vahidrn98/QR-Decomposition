import math
import numpy as np
def zeros(m,n):

    return [[0 for i in range(n)] for j in range(m)]

def zeros_single(m):

    return [0 for j in range(m)]

def norm(A):

    return math.sqrt(sum([pow(x,2) for x in A]))

def transpose(A,m,n):
    
    return [[A[i][j] for i in range(m)] for j in range(n)]

def column(A,i):

    return [row[i] for row in A]

def vec_mult(A,B):

    return sum([A[i]*B[i] for i in range(len(A))])

def mult(A,x):

    return [x*A[i] for i in range(len(A))]

def mult_matrix(A,B):

    return [vec_mult(A[i],B) for i in range(len(A))]

def mat_mult(A,B):

    return [[vec_mult(A[i],column(B,j)) for i in range(len(A))] for j in range(len(B[0]))]

def dot(A,B):

    return [A[i]*B[i] for i in range(len(A))]

def main():

    with open('in.txt','r') as f:

        n_cases = int(f.readline())
        for case in range(n_cases):
            print("test case # ",case+1)
            n = int(f.readline())
            m = int(f.readline())
            # print("(",n,",",m,")")
            A = zeros(n,m)
            y = zeros_single(n)
            x = zeros_single(m)
            Q_t = zeros(m,n)

            R = zeros(m,m)
            v = zeros_single(n)

            for i in range(n):
                for j in range(m):
                    A[i][j] = float(f.readline())

            for i in range(n):
                y[i] = float(f.readline())

            
            for k in range(m):
                v = column(A,k)
                tmp_v =np.copy(v)

                for i in range(k):
                    
                    R[i][k] = vec_mult(Q_t[i],v)
                    
                    t = Q_t[i]
                    
                    tmp_v -= np.dot(np.dot(Q_t[i],v), Q_t[i])

                R[k][k] = norm(tmp_v)

                
                Q_t[k] = [x/norm(tmp_v) for x in tmp_v]
                
            

            Q = transpose(Q_t,m,n)
            
            ##### back substitution #####
           
            b = mult_matrix(Q_t,y)

            
            with open('output.txt','a') as o:

                for k in  range(m-1,-1,-1):
                    tmp = b[k]
                    for i in range(k+1,m):
                        
                        tmp -= x[i] * R[k][i]
                        
                    x[k] = tmp/R[k][k]
                error = norm(np.subtract(mult_matrix(A,x),y))
                print(error)
                if(error<1e-6):
                    o.write(str(error)+'\n')
                else:
                    print("no answer")
                    o.write(str("N")+'\n')
            

            
main()

