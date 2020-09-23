import numpy as np
import numpy.fft as nf
import matplotlib.pyplot as py
import sys
#alpha = 2000        # moderate bandwidth constraint 惩罚因子
#tau = 0             # noise-tolerance (no strict fidelity enforcement) 0 噪声容忍度
#K =5                # VMD分解个数，这里需要人工设置
#DC = 0              # no DC part imposed
#init = 1            # initialize omegas uniformly中心频率初始化
#tol = 1e-10
def cal(alpha,tau,K,DC,init,tol,file,path):
    signal=[]
    with open(file, 'r') as f:#with语句自动调用close()方法
        line = f.readline()
        while line:
            eachline = line###按行读取文本文件，每行数据以列表形式返回
            signal.append(eachline.strip())
            line = f.readline()
    Fs=2000
    save_T = len(signal)
    fs = 1/save_T
    #extend the signal by mirroring
    T = save_T
    #print(signal[0,:T//2] )

    f_mirror = []
    if(save_T %2==0):
        temp = signal[0:T//2]
        f_mirror.extend(temp[::-1]) #temp[::-1] 倒序排列
        f_mirror.extend(signal)
        temp = signal[T//2:T]
        f_mirror.extend(temp[::-1])
    else:
        temp = signal[1:T//2+1]
        f_mirror.extend(temp[::-1]) #temp[::-1] 倒序排列
        f_mirror.extend([0])
        f_mirror.extend(signal)
        temp = signal[T//2+1:T]
        f_mirror.extend(temp[::-1])
    f = f_mirror
    T = np.size(f)
    #t = (1:T)/T;
    t=np.mat(np.zeros((1,T)))
    for i in range(0,T):
        t[0,i]=(i+1)/T
    #freqs = t-0.5-1/T;   
    freqs=np.mat(np.zeros((1,T)))
    for i in range(0,T):
        freqs[0,i]=t[0,i]-0.5-1/T
    N = 500
    Alpha = alpha * np.ones(K)
    #f_hat = fftshift((fft(f)))

    f_hat=np.fft.fftshift(np.fft.fft(f))

    f_hat_plus = f_hat
    f_hat_plus[0:T // 2] = 0
    u_hat_plus=np.zeros((N,np.size(freqs),K),dtype=complex)
    omega_plus = np.zeros((N, K))
    if init==1:
        for i in range(0,K):
            omega_plus[0,i] = (0.5/K)*(i)
    elif init==2:
        omega_plus[0,:] = np.sort(np.exp(np.log(fs) + (np.log(0.5)-np.log(fs))*np.random.rand(1,K)))
    else:
        omega_plus[0,:] = 0

    if DC:
        omega_plus[0,0] = 0

    lambda_hat = np.zeros((N, np.size(freqs)), dtype = complex)
    eps = 2.2204e-16  # python里没有eps功能
    uDiff = tol + eps  # update step
    #uDiff = tol+np.finfo(float).eps
    n = 1
    sum_uk = 0  

    while ( uDiff > tol and  n < N ):
        k = 1
        sum_uk = u_hat_plus[n-1,:,K-1] + sum_uk - u_hat_plus[n-1,:,0]
        u_hat_plus[n,:,k-1]=(f_hat_plus-sum_uk-lambda_hat[n-1,:]/2)/(1+  Alpha[k-1]*np.square(freqs-omega_plus[n-1,k-1]))
        if not(DC):
            omega_plus[n,k-1] = np.dot(freqs[0,T//2:T],(abs(u_hat_plus[n, T//2:T, k-1])**2).T)/np.sum(abs(u_hat_plus[n,T//2:T,k-1])**2)
        for k in range(2,K+1):
            sum_uk=u_hat_plus[n,:,k-2]+sum_uk-u_hat_plus[n-1,:,k-1]
            u_hat_plus[n,:,k-1]=(f_hat_plus-sum_uk-lambda_hat[n-1,:]/2)/(1+Alpha[k-1]*np.square(freqs-omega_plus[n-1,k-1]))
            omega_plus[n,k-1] = np.dot(freqs[0,T//2:T],(abs(u_hat_plus[n, T//2:T, k-1])**2).T)/np.sum(abs(u_hat_plus[n,T//2:T,k-1])**2)
        
        lambda_hat[n,:]=lambda_hat[n-1,:]+tau*(np.sum(u_hat_plus[n,:,:],axis=1)-f_hat_plus)

        n = n+1
        uDiff = eps
        for i in range(1,K+1):
            #uDiff = uDiff + 1/float(T)*np.dot((u_hat_plus[n-1,:,i-1]-u_hat_plus[n-2,:,i-1]),np.conj((u_hat_plus[n-1,:,i-1]-u_hat_plus[n-2,:,i-1])))
            uDiff=uDiff+1/T*np.dot(u_hat_plus[n-1,:,i-1]-u_hat_plus[n-2,:,i-1],(np.conj(u_hat_plus[n-1,:,i-1]-u_hat_plus[n-2,:,i-1])).conj().T)
        uDiff = np.abs(uDiff)   
    N = np.min([N,n])
    omega = omega_plus[:N,:]

    # Signal reconstruction
    u_hat = np.zeros([T, K],dtype = complex)
    u_hat[T//2:T,:] = np.squeeze(u_hat_plus[N-1,T//2:T,:])
    u_hat[T//2:0:-1,:] = np.squeeze(np.conj(u_hat_plus[N-1,T//2:T,:]))
    u_hat[0,:] = np.conj(u_hat[-1,:])   

    u = np.zeros((K,np.size(t)),dtype=complex)
    for k in range(K):
        u[k,:]= np.real(nf.ifft(nf.ifftshift(u_hat[:,k])))

    # remove mirror part 

    if(save_T %2==0):
        u=u[:,T//4:3*T//4]
    else:
        u=u[:,T//4+1:3*T//4+1]
    # print(u_hat.shape)
    #recompute spectrum
    u_hat = np.zeros((T//2,K),dtype=complex)

    for k in range(K):
        u_hat[:,k]=nf.fftshift(nf.fft(u[k,:])).conj().T
    #u, u_hat, omega

    for k in range(K):
        np.savetxt(path+"u"+str(k)+".txt",u[k,:])
        np.savetxt(path+"uabs"+str(k)+".txt",u[k,:].real)

    np.savetxt(path+"u_hat.txt",u_hat)
    np.savetxt(path+"omega.txt",omega)

    c=np.zeros((K,save_T),dtype=complex)
    N=save_T
    for i in range(K):
        f=[]
        c[i,:]=np.fft.fft(u[i,:])
        mag=np.abs(c[i,:]);  
        for j in range(np.size(c[i,:])):
            f.append(Fs*j/np.size(c[i,:]))
        py.subplot(K,1,i+1)
        py.plot(f[0:N//2],mag[0:N//2]*2/N)
        np.savetxt(path+"magx_"+str(i)+".txt",f[0:N//2])
        np.savetxt(path+"magy_"+str(i)+".txt",mag[0:N//2]*2/N)
    Y=u[0,:].T
    for i in range(1,K):
        Y+=u[i,:].T
    e=(Y>0)
    Y=np.where(e,Y,0)
    #np.savetxt(path+"lb.txt",np.log(abs(Y.real)))
    np.savetxt(path+"lb.txt",np.log(Y).real)
    print(1)
cal(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]),float(sys.argv[6]),sys.argv[7],sys.argv[8])