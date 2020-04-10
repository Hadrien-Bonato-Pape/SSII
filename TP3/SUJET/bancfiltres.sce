function [s,E,Esignal,fe]=bancfiltres(M,R,fichier, play)
//Exemple [s,e,es,fe]=bancfiltres(8,128,'piano.wav',1);
[e,fe]=wavread(fichier);
N= R/(4*M);
H=[ones(1,N-1),0.9,0.5,0.1,zeros(1,R-2*N-3),...
   0.1,0.5,0.9,ones(1,N-2)];
h=fftshift(real(ifft(H)));
n=0:R-1;
for j=0:M-1
  bande(j+1,:)=2*cos((2*j+1)*n*%pi/(2*M)).*h;
end
for j=0:M-1
  sfiltre=convol(e,bande(j+1,:)); 
  s(j+1,:)=sfiltre(1:length(e));
  wavwrite(s(j+1,:),fe,['s'+string(j+1)+'.wav']);
end
Esignal= e*e'/2; 
E=diag(s*s')/2;
disp(['sum(E):',string(sum(E))])
bar([0:M-1]*fe/(2*M),100*E/Esignal)
xtitle(['Analyse de ',fichier],'frequence (Hz)'...
        ,'energie (% energie totale)')
xgrid();
if play then sound(sum(s,1),fe);
end
endfunction

