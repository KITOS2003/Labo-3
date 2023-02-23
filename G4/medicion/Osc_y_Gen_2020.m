 %% Osc_y_Gen - Ilustra la diferencia entre acoplar los canales en DC o AC
 
 % Osciloscopio: Tektronix TBS1052B-EDU
 % Generador: Tektronix AFG1022
 
 % Cesar Moreno - DF FCEN UBA e INFIP CONICET
 % Laboratorio 3 - 2do Cuatrimestre 2019

%% Acciones previas
clear all       % Borro variables previas
pause on        % Habilito la funcion PAUSE (permite hacer pausas en la 
                % ejecucion del programa

%% Habilito y verifico la comunicacion con el Osciloscopio Tektronix TBS1052B-EDU
 
if(exist('osc') == 1)
    % Si el canal de comunicacion ya esta abierto, NO LO ABRO NUEVAMENTE.
    disp('La comunicacion con el osciloscopio ya esta abierta')
else
    % Si el canal de comunicacion aun no esta abiero, lo DEFINO ahora. 
    % Uso un puerto USB de la PC y el protocolo denominado: 
    % Virtual Instrument Software Architecture (VISA). 
    
    osc = visa('ni','USB0::0x0699::0x0368::C017041::INSTR')
    
    % Se necesita incrementar el tama#o del buffer de comunicacion para 
    % poder recabar/adquirir señales extensas.
    
    set(osc,'InputBufferSize',20000)
    
    % Finalmente ABRO la comunicacion con el instrumento.
    
    fopen(osc);
end

% Ahora pruebo si la communicacion funciona correctamente

% Le pido al osciloscopio algo elemental: que se identifique. 
% Si da error, la comunicacion no esta correctamente establecida y carece 
%   de sentido continuar ejecutando el resto del programa.

fprintf(osc,'*IDN?')    % Pregunta la identificacion.       
ID = fscanf(osc)        % Escribe la identificacion en la pantalla.

%% Habilito y verifico la comunicacion con el Generador de Funciones Tektronix AFG1022

if(exist('gen') == 1)
    % Si el canal de comunicacion ya esta abierto, NO LO ABRO NUEVAMENTE.
    disp('La comunicacion con el generador ya esta abierta')
else
    % Si el canal de comunicacion aun no esta abiero, lo DEFINO ahora. 
    % Uso un puerto USB de la PC y el protocolo denominado: 
    % Virtual Instrument Software Architecture (VISA). 
    
    gen = visa('ni','USB::0x0699::0x0353::1915264::INSTR')
    
    % Finalmente ABRO la comunicacion con el instrumento.
    
    fopen(gen);
end

% Pruebo si la communicacion funciona correctamente

% Le pido al generador algo elemental: que se identifique. 
% Si da error, la comunicacion no esta correctamente establecida y carece 
%   de sentido continuar ejecutando el resto del programa.

fprintf(gen,'*IDN?')    % Pregunta la identificacion.       
ID = fscanf(gen)        % Escribe la identificacion en la pantalla.


%% ESPECIFICO parte de la configuracion del osciloscopio (como ejemplos) 

% El resto debe hacerse manualmente

% ACOPLO el CH1 en DC y el CH2 en AC (CC en castellano)
fprintf(osc,'CH1:COUPling DC')
fprintf(osc,'CH2:COUPling AC')

% Ubico el cero del eje vertical de cada canal a mitad de la pantalla  
fprintf(osc,'CH1:POSition 0.0E0')
fprintf(osc,'CH2:POSition 0.0E0')

% Especifico que la amplificacion vertical sea 2V/div en cada canal
fprintf(osc,'CH1:SCAle 2.0E0')
fprintf(osc,'CH2:SCAle 2.0E0')

% Especifico que el modo de adquisicion sea promediando se#ales
fprintf(osc,'ACQuire:MODe AVErage')

% Promedio un "NroProm" de mediciones, por ej. 16
NroProm = 16;    % Se admiten los valores: 4, 16, 64 o 128
fprintf(osc,['ACQuire:NUMAVg ', num2str(NroProm)])

% Especifico que las "sondas" conectadas a los canales 1 y 2 "dividen" por 1
fprintf(osc,'CH1:PRObe 1')
fprintf(osc,'CH2:PRObe 1')

% Especifico que ningun canal invierta la se#al que le llegue
fprintf(osc,'CH1:INVert OFF')
fprintf(osc,'CH2:INVert OFF')

% TRIGGER. Especifico canal, nivel y pendiente del Trigger
fprintf(osc,'TRIGger:MAIn:EDGE:SOUrce CH1')
fprintf(osc,'TRIGger:MAIn:LEVel 1.0')
fprintf(osc,'TRIGger:MAIn:EDGE:SLOpe RISe')



% ESPECIFICO LAS MEDICIONES AUTOMATICAS QUE NECESITO
% (Borrar las mediciones automaticas preexistentes)

% PROBAR ESTO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf(osc,'MEASUrement:MEAS1:TYPe NONe')
fprintf(osc,'MEASUrement:MEAS2:TYPe NONe')
fprintf(osc,'MEASUrement:MEAS3:TYPe NONe')
fprintf(osc,'MEASUrement:MEAS4:TYPe NONe')
fprintf(osc,'MEASUrement:MEAS5:TYPe NONe')
fprintf(osc,'MEASUrement:MEAS6:TYPe NONe')
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Eespecifico que la:
%   Medicion 1 sea la frecuencia de la se#al enviada al CH1
fprintf(osc,'MEASU:MEAS1:SOURCE CH1') 
fprintf(osc,'MEASU:MEAS1:TYPE FREQuency')
%   Medicion 2 sea el valor pico a pico del canal 1
fprintf(osc,'MEASU:MEAS2:SOUrce CH1') 
fprintf(osc,'MEASU:MEAS2:TYPE PK2pk')
%   Medicion 3 sea el valor pico a pico del canal 2
fprintf(osc,'MEASU:MEAS3:SOUrce CH2') 
fprintf(osc,'MEASU:MEAS3:TYPE PK2pk')


%% Preparo la salida 1 de la fuente
% Especifico el tipo de se#al, amplitud, offset y luego HABILITO la salida
fprintf(gen,'SOURce1:FUNCtion:SHAPe SINusoid')
fprintf(gen,'SOURce1:VOLTage 10 Vpp')     
fprintf(gen,'SOURce1:VOLTage:OFFSet 0')         
fprintf(gen,'OUTPut1:STATe ON')

%% Lazo de medicion

% Especifico el rango de frecuencias a explorar, EXPRESADAS EN Hz
%f = [logspace(0,2,9) logspace(log10(200),4,5)];
f = [3 5 10 15 20 30 40 50 100 200];

% Defino la matriz de SALIDA como nula (vacia). 
%   Luego contendra las 4 columnas siguientes: 
%   1) frecuencia  2) Vpp_del_CH1  3) Vpp_del_CH2 4) Vpp_del_Mult
salida = [];
salidanueva = []; % Defino una salida parcial auxiliar

for i = 1:length(f)
disp(['Indice de Medicion:',num2str(i),' Frec Nominal en Hz: ',num2str(f(i))] )
fprintf(gen,['SOUR1:FREQ ',num2str(f(i))])           

% Especifico velocidad de barrido horizontal (seg/div) de modo de tener
% unos 5 periodos en la pantalla, aproximadamente.

fprintf(osc,['HORizontal:MAIn:SCAle ',num2str(1.5/(10*f(i)))]) % En seg/div

disp('Controlar Frecuencia en la Fuente.')
disp('Controlar/CORREGIR el barrido horizontal del Osciloscopio')
disp('Controlar que ninguna de las mediciones tenga un signo: ?')
disp('Colocar el cursor en la VENTANA DE COMANDOS y apretar tecla')
pause 

% Espero a que se complete el "NroProm" de se#ales que le pedi que promedie
disp(['Promediando ',num2str(NroProm),' señales...'])
pause(NroProm * 1.0/(2*f(i)) * 12 )

% Mido la frecuencia
fprintf(osc,'MEASU:MEAS1:VAL?') 
frecuencia = str2num(fscanf(osc))

% Mido el Vpp_del_CH1
fprintf(osc,'MEASU:MEAS2:VAL?') 
Vpp_del_CH1 = str2num(fscanf(osc))

% Mido el Vpp_del_CH2
fprintf(osc,'MEASU:MEAS3:VAL?') 
Vpp_del_CH2 = str2num(fscanf(osc))

% Ingreso manualmente la lectura del multimetro
lectura = input('Ingresar manualmente la lectura del mult. y apretar ENTER');
Vpp_del_Mult = 2 * sqrt(2) * lectura;

salidanueva = [frecuencia Vpp_del_CH1 Vpp_del_CH2 Vpp_del_Mult]

% Voy construyendo la matriz de salida
salida = [salida; salidanueva];

% Grafico lo obtenido hasta ahora
figure(1)
plot(salida(:,1),salida(:,2),'*-', ...
    salida(:,1),salida(:,3),'sr-', ...
    salida(:,1),salida(:,4),'cr-'))
xlabel('f (Hz)')
ylabel('Vpp de los canales 1 (DC) y 2 (AC)')
title('Comparación entre acoplamiento del Osc. en DC o AC')
grid on
drawnow

disp('Colocar el cursor en la pantalla de comandos y apretar tecla')
pause


end

%% Guardo las mediciones en los archivos: mediciones.txt y mediciones.csv
save mediciones.txt salida -ASCII
% Y tambien en CommaSeparatedValues (.csv) . Compatible con Python, Origin y Excel
csvwrite('mediciones.csv', salida)


%% Grafico final
figure(2)
plot(salida(:,1),salida(:,2),'*-',salida(:,1),salida(:,3),'sr-')
xlabel('f (Hz)')
ylabel('Vpp de los canales 1 (DC) y 2 (AC)')
title('Comparación entre acoplamiento del Osc. en DC o AC')
grid on 

%% Finalizo
%fprintf(osc,'*CLS')

pause off


