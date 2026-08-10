
from odbAccess import *
from numpy import *
import sys
import os

filename = 'new.odb'
session.upgradeOdb(existingOdbPath='TUX', upgradedOdbPath='new')

odb = openOdb(path='new.odb')
myAssembly = odb.rootAssembly

#Number of nonzero components of stress, strain
ntens=6

#print 'Node sets = ',odb.rootAssembly.instances['PART-1-1'].nodeSets.keys()
#print 'Element sets = ',odb.rootAssembly.instances['PART-1-1'].elementSets.keys()
#for stepName in odb.steps.keys():
#      print stepName

#lastFrame = odb.steps['Step-1'].frames[-1]
#print lastFrame

#for fieldName in lastFrame.fieldOutputs.keys():
#      print fieldName


SALLE = odb.rootAssembly.instances['PART-1-1'].elementSets['ALLE']
URIGHT = odb.rootAssembly.instances['PART-1-1'].nodeSets['RIGHT']
UTOP = odb.rootAssembly.instances['PART-1-1'].nodeSets['TOP']
UFRONT = odb.rootAssembly.instances['PART-1-1'].nodeSets['FRONT']

#print odb.steps['Step-1'].frames[-1].fieldOutputs.keys()

#print len(odb.steps['Step-1'].frames)
#sys.exit()

aveFile = open('averages.dat','w')
#aveFile2 = open('averages2.dat','w')

F=zeros((4,4),dtype=float)
Finv=zeros((4,4),dtype=float)
SCb=zeros((4,4),dtype=float)
Sb=zeros((4,4),dtype=float)


for i in range(0,len(odb.steps['Step-1'].frames)) :
#for i in range(1,4) :
    stressField = odb.steps['Step-1'].frames[i].fieldOutputs['S']
    sfield = stressField.getSubset(region=SALLE,position=CENTROID)

    volField = odb.steps['Step-1'].frames[i].fieldOutputs['EVOL']
    vfield = volField.getSubset(region=SALLE,position=WHOLE_ELEMENT)

    totnumElems=len(sfield.values)

    if i==0 :
        print
        print 'Total Number of elements=', totnumElems
        print

    s=zeros(ntens,dtype=float)
    for j in range(0, totnumElems):
        vol = vfield.values[j]
        for k in range(0,ntens):
              s[k] = s[k] + vol.data*sfield.values[j].data[k]

    # Compute the Cauchy stress averages   
    SCb[1,1]=s[0]
    SCb[2,2]=s[1]
    SCb[3,3]=s[2]
    SCb[1,2]=s[3]
    SCb[2,1]=s[3]
    SCb[1,3]=s[4]
    SCb[3,1]=s[4]      
    SCb[2,3]=s[5]
    SCb[3,2]=s[5]

    # Compute the Average Deformation Gradient (from applied nodal displacement)
    displacement= odb.steps['Step-1'].frames[i].fieldOutputs['U']
    rightDisplacement = displacement.getSubset(region=URIGHT)
    topDisplacement = displacement.getSubset(region=UTOP)
    frontDisplacement = displacement.getSubset(region=UFRONT)

    for k in range(1,ntens/2+1):
        F[k,1]=rightDisplacement.values[0].data[k-1]
        F[k,2]=topDisplacement.values[0].data[k-1]
        F[k,3]=frontDisplacement.values[0].data[k-1]

    F[1,1]=F[1,1]+1
    F[2,2]=F[2,2]+1
    F[3,3]=F[3,3]+1

        
    detF=F[1,3]*(F[2,1]*F[3,2]-F[2,2]*F[3,1])+F[1,2]*(F[2,3]*F[3,1]-F[2,1]*F[3,3])+F[1,1]*(F[2,2]*F[3,3]-F[2,3]*F[3,2])
    Finv[1,1]=(-F[2,3]*F[3,2]+F[2,2]*F[3,3])/detF
    Finv[2,1]=(-F[2,1]*F[3,3]+F[2,3]*F[3,1])/detF
    Finv[3,1]=(-F[2,2]*F[3,1]+F[2,1]*F[3,2])/detF
    Finv[1,2]=(-F[1,2]*F[3,3]+F[1,3]*F[3,2])/detF
    Finv[2,2]=(-F[1,3]*F[3,1]+F[1,1]*F[3,3])/detF
    Finv[3,2]=(-F[1,1]*F[3,2]+F[1,2]*F[3,1])/detF
    Finv[1,3]=(-F[1,3]*F[2,2]+F[1,2]*F[2,3])/detF
    Finv[2,3]=(-F[1,1]*F[2,3]+F[1,3]*F[2,1])/detF
    Finv[3,3]=(-F[1,2]*F[2,1]+F[1,1]*F[2,2])/detF

    for ii in range(1,ntens/2+1):
        for jj in range(1,ntens/2+1):                  
              Sb[ii,jj]=0.0
              for k in range(1,ntens/2+1):
                     Sb[ii,jj]=Sb[ii,jj]+SCb[ii,k]*Finv[jj,k]*detF

 

    aveFile.write('%12.8E %12.8E %12.8E %12.8E %12.8E %12.8E %12.8E %12.8E %12.8E %12.8E %12.8E %12.8E %12.8E %12.8E %12.8E %12.8E\n' % (F[1,1], F[2,2], F[3,3],F[1,2], F[1,3], F[2,3], Sb[1,1],Sb[1,2],Sb[1,3],Sb[2,1],Sb[2,2],Sb[2,3],Sb[3,1],Sb[3,2],Sb[3,3],detF))

#    aveFile2.write('%12.8E %12.8E %12.8E %12.8E %12.8E %12.8E %12.8E %12.8E %12.8E %12.8E %12.8E %12.8E %12.8E %12.8E %12.8E %12.8E %12.8E %12.8E\n' % (F[1,1], F[1,2], F[1,3], F[2,1], F[2,2], F[2,3], F[3,1], F[3,2],F[3,3], SCb[1,1],SCb[1,2],SCb[1,3],SCb[2,1],SCb[2,2],SCb[2,3],SCb[3,1],SCb[3,2],SCb[3,3]))
      


    

#      print 'frame=', i
#      print 'Sb11=', s11ave
#      print 'frame',i,'  ','F22=', '%-12.8f' % F[2,2],'  ','SCb22=', '%-12.8f' % SCb[2,2],'  ','Sb22=', '%-12.8f' % Sb[2,2]
#      print 'Sb33=', s33ave
#      print 'Sb12=', s12ave
#      print 'Sb13=', s13ave
#      print 'Sb23=', s23ave

aveFile.close()   
# aveFile2.close()   
session.odbs['new.odb'].close()

##### delete temporal upgraded file
os.remove("new.odb") 