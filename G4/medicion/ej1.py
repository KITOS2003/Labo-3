# Osciloscopio Tektronix TBS 1052B-EDU
#
# Adquiere los datos de los canales 1 y/o 2.
# Haciendo uso de un puerto USB de la PC y el protocolo denominado: 
# VISA (Virtual Instrument Software Architecture). 
#
#
# Laboratorio 3, 2do Cuatrimestre 2018, DF, FCEN, UBA.
# Cesar Moreno

import numpy as np
import pyvisa as visa
import matplotlib.pyplot as plt

def AdqTek(osci, CHx):
	''' 
	Funcion para adquirir datos de algun canal del OSC

    Input:
		osci - IO STREAM - Objeto de conexion con el osciloscopio
		CHx - INT or STR - Canal del que se quiere adquirir los datos
    
    Output:
		t - Numpy array - Tiempos del canal medido
		V - Numpy array - Valores de tension del canal medido
	'''
	
	assert str(CHx) in ('1', '2'), "El canal elegido es incorrecto"

	osci.write('DATA:SOURCE CH'+str(CHx)) # Le digo que canal quiero que adquiera

	# Modo de transmision: Binario (osea, la onda digitalizada)
	data = osci.query_binary_values('CURV?', datatype='B', container=np.array) 
	osci.write('DAT:ENC RPB') 
	osci.write('DAT:WID 1') 

	# Para poder traducir los datos que devuelve el osc a tensiones (V), necesito ciertas ctes de escala. 
	# Tambien necesito ciertas ctes para construir el eje temporal (t). 
	# Para esto le pido (le hago una QUERY) al osc que me algunos de los datos de formateo, 
	# los WaveForM Preambles (WFMPre)
	XZE, XIN, YZE, YMU, YOFF = osci.query_ascii_values('WFMPre:XZE?;XIN?;YZE?;YMU?;YOFF?;', separator=';')
	# XZEro >> Me dice cual es el instante correspondiente al 1er dato.  
	# XINcr >> Me dice cual es el paso de muestreo temporal. 
	# YZEro >> Me dice cuanto debo adicionarle a Y para obtener V.
	# YMUlt >> Me dice el factor de escala vertical (Amplificacion).
	# YOFFs >> Me dice cuanto vale el corrimiento vertical (Offset).

	# Le pido al osc que me devuelva los valores de la CURVa digitalizados (de alli el binary_values)
	# Y ya que esta que lo empaquete en un array de numpy
	data = osci.query_binary_values('CURV?', datatype='B', container=np.array) 

	# Para recuperar los correctos valores de tension registrados uso los parametros de escala que traje
	V = (data - YOFF)*YMU + YZE;

	#Construyo vector de tiempos con los parametros que traje
	XEND = XZE + XIN*len(V);
	t = np.arange(XZE, XEND, XIN)
	return t, V



#Primero verificar que efectivamente esta bien conectado el oscilosopio
rm = visa.ResourceManager()
instrumentos = rm.list_resources()  
# Esto lista todos los dispositivos disponibles, uno de los cuales
# deberia ser "USB0::0x0699::0x0368::C017044::INSTR", donde los terminos
# indican "puerto::marca::modelo::nro_de_serie" del instrumento.

# Con ese nombre abro el vinculo con el osciloscopio
with rm.open_resource(instrumentos) as osc:

	'''	
	Para interactuar con el osciloscopio hay dos formas basicas, una es pedirle informacion
	lo que quiere decir hacerle una pregunta (query) al mismo. Para esto bastaria utilizar
		
	osc.query("Lo_que_quiero_preguntarle_al_osc")
	Ejemplo: osc.query("*IDN?") << Pregunta como se identifica el osciloscopio
	 
	Otra cosa que se podria querer hacer es SETEAR algun parametro, o pedirle al osc que
	realice alguna operacion, para esto usar:
	
	osc.write("Lo_que_le_pido_al_osc")  #Analogo al fprintf de MATLAB
	Ejemplo: osc.write("AUTOSet EXECute") << Realiza una autoconfiguracion de los canales

	Para saber que se le puede preguntar o pedir al osciloscopio hay que leer y comprender 
	el manual del programador del instrumento.
	'''
	

	# Ahora le pido al osciloscopio algo elemental: que se identifique. 
	# Si da error, la comunicacion no esta correctamente establecida y carece 
	# de sentido continuar ejecutando el resto del programa.
	print(osc.query('*IDN?'))    # Pregunta la identificacion y la escribe en pantalla.

	tiempo1, v1 = AdqTek(osc, 1) # Lee el CH1
	tiempo2, v2 = AdqTek(osc, 2) # Lee el CH2

	# Hago una visualizacion de los valores obtenidos
	plt.figure() # Creo una figura para el grafico
	plt.plot(tiempo1, v1) # Agrego los puntos del canal 1 que quiero graficar
	plt.plot(tiempo2, v2) # Agrego los puntos del canal 2 que quiero graficar
	plt.grid(True) # Pongo grillas en ambos ejes
