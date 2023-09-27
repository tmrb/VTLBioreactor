import numpy as np
import matplotlib.pyplot as plt

# Parâmetros do reator
N_riser = 3  # Número de volumes na seção de subida (riser)
N_gas_separator = 3  # Número de volumes na seção de separação de gás
N_downcomer = 3  # Número de volumes na seção de descida (downcomer)
comprimento_riser = 0.4  # Comprimento da seção de subida
comprimento_gas_separator = 0.2  # Comprimento da seção de separação de gás
comprimento_downcomer = 0.4  # Comprimento da seção de descida
comprimento_reator = comprimento_riser + comprimento_gas_separator + comprimento_downcomer

# Número total de volumes
N = N_riser + N_gas_separator + N_downcomer

# Cálculo do volume de cada seção
volume_riser = comprimento_riser / N_riser
volume_gas_separator = comprimento_gas_separator / N_gas_separator
volume_downcomer = comprimento_downcomer / N_downcomer

# Vetores de posição para cada seção
posicoes_riser = np.linspace(0, comprimento_riser, N_riser + 1)
posicoes_gas_separator = np.linspace(comprimento_riser, comprimento_riser + comprimento_gas_separator, N_gas_separator + 1)
posicoes_downcomer = np.linspace(comprimento_riser + comprimento_gas_separator, comprimento_reator, N_downcomer + 1)

# Condições iniciais e parâmetros (valores fictícios, ajuste conforme necessário)
C_metano_riser = np.zeros(N_riser + 1)  
C_metano_gas_separator = np.zeros(N_gas_separator + 1) 
C_metano_downcomer = np.zeros(N_downcomer + 1)  
C_oxigenio_riser = np.zeros(N_riser + 1)  
C_oxigenio_gas_separator = np.zeros(N_gas_separator + 1)  
C_oxigenio_downcomer = np.zeros(N_downcomer + 1)  
C_biomassa_riser = np.zeros(N_riser + 1)  
C_biomassa_gas_separator = np.zeros(N_gas_separator + 1)  
C_biomassa_downcomer = np.zeros(N_downcomer + 1)  
C_produto_riser = np.zeros(N_riser + 1)  
C_produto_gas_separator = np.zeros(N_gas_separator + 1) 
C_produto_downcomer = np.zeros(N_downcomer + 1)  
KLaO = 0.0121  # Coeficiente de transferência de massa de oxigênio no riser
KLa_gas_separator = 0.005  # Coeficiente de transferência de massa de oxigênio na seção de separação de gás
KLaM = 0.0104  # Coeficiente de transferência de massa de oxigênio na descida
mu_max = 0.212  # Taxa máxima de crescimento da biomassa
Yxs = 101.54  # Coeficiente de conversão de substrato em biomassa
Yxo = 1.0002 # Coeficiente de conversão de oxigênio em biomassa
Yp = 0.4  # Coeficiente de conversão de substrato em produto
gas_holdup_riser = 0.324  # Gas hold-up no riser
gas_holdup_gas_separator = 0.2  # Gas hold-up na seção de separação de gás
gas_holdup_downcomer = 0.189  # Gas hold-up na descida
Ksm = 0.00058
Kso = 0.00003
Kd = 0.322
Kl = 0.00025
Ai_riser = np.pi*(0.3)**2
Li_riser = 2.20
Ai_gas = np.pi*(0.9)**2
Li_gas = 0.10
Ai_down = np.pi*(0.15)**2
Li_down = 2.20
Vol_total = Ai_riser*Li_riser + Ai_gas*Li_gas + Ai_down*Li_down
print(Vol_total)
KLa_riser = 0.1  # Coeficiente de transferência de massa de oxigênio no riser
KLa_gas_separator = 0.05  # Coeficiente de transferência de massa de oxigênio na seção de separação de gás
KLa_downcomer = 0.08  # Coeficiente de transferência de massa de oxigênio na descida

QL = 1.2*60 #vazão de líquido 2 a 20 L/h
# Setando as concentrações iniciais em cada seção
C_metano_riser[:] = 10.0  # Concentração inicial no primeiro volume do riser
C_oxigenio_riser[:] = 2  # Concentração inicial de oxigênio no riser
C_biomassa_riser[:] = .1  # Concentração inicial de biomassa no riser
C_metano_gas_separator[:] = 10.0  # Concentração inicial no primeiro volume do riser
C_oxigenio_gas_separator[:] = 2  # Concentração inicial de oxigênio no riser
C_biomassa_gas_separator[:] = .1  # Concentração inicial de biomassa no riser [micromol/L]
C_metano_downcomer[:] = 10.0  # Concentração inicial no primeiro volume do riser
C_oxigenio_downcomer[:] = 2  # Concentração inicial de oxigênio no riser
C_biomassa_downcomer[:] = 0.1  # Concentração inicial de biomassa no riser
C_metano_sat = 0.04812
C_oxigenio_sat = 2.55
# Vetor de tempo

tempo_total = 40
t = np.linspace(0, tempo_total, 100)

# Arrays para armazenar resultados
resultados_metano_riser = np.zeros((N_riser + 1, len(t)))
resultados_metano_gas_separator = np.zeros((N_gas_separator + 1, len(t)))
resultados_metano_downcomer = np.zeros((N_downcomer + 1, len(t)))
resultados_oxigenio_riser = np.zeros((N_riser + 1, len(t)))
resultados_oxigenio_gas_separator = np.zeros((N_gas_separator + 1, len(t)))
resultados_oxigenio_downcomer = np.zeros((N_downcomer + 1, len(t)))
resultados_biomassa_riser = np.zeros((N_riser + 1, len(t)))
resultados_biomassa_gas_separator = np.zeros((N_gas_separator + 1, len(t)))
resultados_biomassa_downcomer = np.zeros((N_downcomer + 1, len(t)))
resultados_produto_riser = np.zeros((N_riser + 1, len(t)))
resultados_produto_gas_separator = np.zeros((N_gas_separator + 1, len(t)))
resultados_produto_downcomer = np.zeros((N_downcomer + 1, len(t)))

# Loop de tempo
for idx, tempo in enumerate(t):
    for i in range(N+1):
        if i < N_riser+1:
            # Calcular taxas de reação e transferência de massa no riser
            taxa_metano = (-1/Yxs)* (C_metano_riser[i] - C_metano_gas_separator[0])
            taxa_oxigenio = (-1/Yxo)* (C_oxigenio_riser[i] - C_oxigenio_gas_separator[0])
            taxa_biomassa = mu_max * (C_metano_riser[i] / (C_metano_riser[i] + Ksm)) * (C_oxigenio_riser[i] / (C_oxigenio_riser[i] + Kso)) * C_biomassa_riser[i] - Kd*C_biomassa_riser[i] 
            
            C_metano_riser[i] += ((QL*0.001)/((Ai_riser*Li_riser)*(1-gas_holdup_riser)))*(C_metano_riser[i-1] - C_metano_riser[i]) + taxa_metano*volume_riser  + KLaM*(C_metano_sat - C_metano_riser[i])
            C_oxigenio_riser[i] += (((1/4)*QL*0.001)/((Ai_riser*Li_riser)*(1-gas_holdup_riser)))*(C_oxigenio_riser[i - 1] - C_oxigenio_riser[i]) + taxa_oxigenio*volume_riser + KLaO*(C_oxigenio_sat - C_oxigenio_riser[i])
            C_biomassa_riser[i] += ((QL*0.01)/((Ai_riser*Li_riser)*(1-gas_holdup_riser)))*(C_metano_riser[i - 1] - C_metano_riser[i]) + taxa_biomassa*volume_riser 
            C_produto_riser[i] += Yp *C_biomassa_riser[i]
            
            # Armazenar resultados
            resultados_metano_riser[i, idx] = C_metano_riser[i]
            #resultados_metano_gas_separator[i, idx] = C_metano_gas_separator[i]
            resultados_oxigenio_riser[i, idx] = C_oxigenio_riser[i]
            #resultados_oxigenio_gas_separator[i, idx] = C_oxigenio_gas_separator[i]
            resultados_biomassa_riser[i, idx] = C_biomassa_riser[i]
            #resultados_biomassa_gas_separator[i, idx] = C_biomassa_gas_separator[i]
            resultados_produto_riser[i, idx] = C_produto_riser[i]

        if  N_riser -1 < i <= N_riser + N_gas_separator:
            # Calcular taxas de reação e transferência de massa na seção de separação de gás
            taxa_metano = (-1/Yxs)* (C_metano_gas_separator[i - N_riser] - C_metano_downcomer[0])
            taxa_oxigenio = (-1/Yxo)* (C_oxigenio_gas_separator[i - N_riser]- C_oxigenio_downcomer[0]) 
            taxa_biomassa = mu_max * (C_metano_gas_separator[i - N_riser] / (C_metano_gas_separator[i - N_riser] + Ksm)) * (C_oxigenio_gas_separator[i - N_riser] / (C_oxigenio_gas_separator[i - N_riser] + Kso)) * C_biomassa_gas_separator[i - N_riser] - Kd*C_biomassa_gas_separator[i - N_riser]
            
            C_metano_gas_separator[i - N_riser] += ((QL*0.001)/((Ai_gas*Li_gas)*(1-gas_holdup_gas_separator)))*(C_metano_gas_separator[i - N_riser - 1] - C_metano_gas_separator[i - N_riser]) + taxa_metano*volume_gas_separator  + KLaM*(C_metano_sat - C_metano_gas_separator[i - N_riser])
            C_oxigenio_gas_separator[i - N_riser] += (((1/4)*QL*0.001)/((Ai_gas*Li_gas)*(1-gas_holdup_gas_separator)))*(C_oxigenio_gas_separator[i - N_riser - 1] - C_oxigenio_gas_separator[i - N_riser]) + taxa_oxigenio*volume_gas_separator + KLaO*(C_oxigenio_sat - C_oxigenio_gas_separator[i - N_riser])
            C_biomassa_gas_separator[i - N_riser] += ((QL*0.01)/((Ai_gas*Li_gas)*(1-gas_holdup_gas_separator)))*(C_metano_gas_separator[i - N_riser - 1] - C_metano_gas_separator[i - N_riser]) + taxa_biomassa*volume_gas_separator
            C_produto_gas_separator[i - N_riser] += Yp * C_biomassa_gas_separator[i - N_riser]
            
            
            # Armazenar resultados
            resultados_metano_gas_separator[i - N_riser, idx] = C_metano_gas_separator[i - N_riser]
            #resultados_metano_downcomer[i - N_riser, idx] = C_metano_downcomer[i - N_riser]
            resultados_oxigenio_gas_separator[i - N_riser, idx] = C_oxigenio_gas_separator[i - N_riser]
            #resultados_oxigenio_downcomer[i - N_riser, idx] = C_oxigenio_downcomer[i - N_riser]
            resultados_biomassa_gas_separator[i - N_riser, idx] = C_biomassa_gas_separator[i - N_riser]
            #resultados_biomassa_downcomer[i - N_riser, idx] = C_biomassa_downcomer[i - N_riser]
            resultados_produto_gas_separator[i - N_riser, idx] = C_produto_gas_separator[i - N_riser]
            #resultados_produto_downcomer[i - N_riser, idx] = C_produto_downcomer[i - N_riser]
        
        if i > N_riser + N_gas_separator -1 :
            # Calcular taxas de reação e transferência de massa na descida
            taxa_metano = (-1/Yxs)* (C_metano_downcomer[i - N_riser - N_gas_separator])
            taxa_oxigenio = (-1/Yxo)* (C_oxigenio_downcomer[i - N_riser - N_gas_separator])
            taxa_biomassa = mu_max * (C_metano_downcomer[i - N_riser - N_gas_separator] / (C_metano_downcomer[i - N_riser - N_gas_separator] + Ksm)) * (C_oxigenio_downcomer[i - N_riser - N_gas_separator] / (C_oxigenio_downcomer[i - N_riser - N_gas_separator] + Kso)) * C_biomassa_downcomer[i - N_riser - N_gas_separator] - Kd*C_biomassa_downcomer[i - N_riser - N_gas_separator]
            
            C_metano_downcomer[i - N_riser - N_gas_separator] += ((QL*0.001)/((Ai_down*Li_down)*(1-gas_holdup_downcomer)))*(C_metano_downcomer[i - N_riser - N_gas_separator - 1] - C_metano_downcomer[i - N_riser - N_gas_separator]) + taxa_metano * volume_downcomer + KLaM*0.34*(C_metano_sat - C_metano_downcomer[i - N_riser - N_gas_separator])
            C_oxigenio_downcomer[i - N_riser - N_gas_separator] += (((1/4)*QL*0.001)/((Ai_down*Li_down)*(1-gas_holdup_downcomer)))*(C_oxigenio_downcomer[i - N_riser - N_gas_separator - 1] - C_oxigenio_downcomer[i - N_riser - N_gas_separator]) + taxa_oxigenio * volume_downcomer + KLaO*0.34*(C_oxigenio_sat - C_oxigenio_downcomer[i - N_riser - N_gas_separator])
            C_biomassa_downcomer[i - N_riser - N_gas_separator] += ((QL*0.01)/((Ai_down*Li_down)*(1-gas_holdup_downcomer)))*(C_metano_downcomer[i - N_riser - N_gas_separator - 1] - C_metano_downcomer[i - N_riser - N_gas_separator]) + taxa_biomassa * volume_downcomer 
            C_produto_downcomer[i - N_riser - N_gas_separator] += Yp * C_biomassa_downcomer[i - N_riser - N_gas_separator]
            # Armazenar resultados
            resultados_metano_downcomer[i - N_riser - N_gas_separator, idx] = C_metano_downcomer[i - N_riser - N_gas_separator]
            resultados_oxigenio_downcomer[i - N_riser - N_gas_separator, idx] = C_oxigenio_downcomer[i - N_riser - N_gas_separator]
            resultados_biomassa_downcomer[i - N_riser - N_gas_separator, idx] = C_biomassa_downcomer[i - N_riser - N_gas_separator]
            resultados_produto_downcomer[i - N_riser - N_gas_separator, idx] = C_produto_downcomer[i - N_riser - N_gas_separator]

# Cálculo da quantidade total de PHB produzido
quantidade_total_phb = np.sum(resultados_produto_riser)*volume_riser + np.sum(resultados_produto_gas_separator)*volume_gas_separator + np.sum(resultados_produto_downcomer)*volume_downcomer

# Plotar os resultados
plt.figure(figsize=(2, 8))

#  Metano no Riser
plt.subplot(4, 1, 1)
for i in range(N_riser + 1):
    plt.plot(t, resultados_metano_riser[i], label=f'Volume {i} (Riser)')
plt.xlabel('Time [h]')
plt.ylabel('Methane [mg/L]')
plt.legend(loc='upper center', bbox_to_anchor=(0.7, 1.2),
          fancybox=True, shadow=True, ncol=1)
#  Oxigênio no Riser
plt.subplot(4, 1, 2)
for i in range(N_riser + 1):
    plt.plot(t, resultados_oxigenio_riser[i], label=f'Volume {i} (Riser)')
plt.xlabel('Time [h]')
plt.ylabel(' Oxygen [mg/L]')
#plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))

#  Biomassa no Riser
plt.subplot(4, 1, 3)
for i in range(N_riser + 1):
    plt.plot(t, resultados_biomassa_riser[i], label=f'Volume {i} (Riser)')
plt.xlabel('Time [h]')
plt.ylabel(' Biomass  [mg/L]')

#  Produto (PHB) no Riser
plt.subplot(4, 1, 4)
for i in range(N_riser + 1):
    plt.plot(t, resultados_produto_riser[i], label=f'Volume {i} (Riser)')
plt.xlabel('Time [h]')
plt.ylabel(' PHB [mg/L]')

plt.show()

plt.figure(figsize=(8, 12))
#  Metano na Seção de Separação de Gás
plt.subplot(4, 1, 1)
for i in range(N_gas_separator + 1):
    plt.plot(t, resultados_metano_gas_separator[i], label=f'Volume {i} (Gas Separator)')
plt.xlabel('Time [h]')
plt.ylabel(' Methane [g/L]')
plt.legend(loc='upper center', bbox_to_anchor=(0.7, 1.2),
          fancybox=True, shadow=True, ncol=1)

#  Oxigênio na Seção de Separação de Gás
plt.subplot(4, 1, 2)
for i in range(N_gas_separator + 1):
    plt.plot(t, resultados_oxigenio_gas_separator[i], label=f'Volume {i} (Gas Separator)')
plt.xlabel('Time [h]')
plt.ylabel(' Oxygen [mg/L]')
#plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))

#  Biomassa na Seção de Separação de Gás
plt.subplot(4, 1, 3)
for i in range(N_gas_separator + 1):
    plt.plot(t, resultados_biomassa_gas_separator[i], label=f'Volume {i} (Gas Separator)')
plt.xlabel('Time [h]')
plt.ylabel(' Biomass [mg/L]')


#  Produto (PHB) na Seção de Separação de Gás
plt.subplot(4, 1, 4)
for i in range(N_gas_separator + 1):
    plt.plot(t, resultados_produto_gas_separator[i], label=f'Volume {i} (Gas Separator)')
plt.xlabel('Time [h]')
plt.ylabel(' PHB  [mg/L]')
plt.show()

plt.figure(figsize=(8, 12))
#  Metano na Descida
plt.subplot(4, 1, 1)
for i in range(N_downcomer + 1):
    plt.plot(t, resultados_metano_downcomer[i], label=f'Volume {i} (Downcomer)')
plt.xlabel('Time [h]')
plt.ylabel(' Methane [mg/L]')
plt.legend(loc='upper center', bbox_to_anchor=(0.7, 1.2),
          fancybox=True, shadow=True, ncol=1)

#  Oxigênio na Descida
plt.subplot(4, 1, 2)
for i in range(N_downcomer + 1):
    plt.plot(t, resultados_oxigenio_downcomer[i], label=f'Volume {i} (Downcomer)')
plt.xlabel('Time [h]')
plt.ylabel(' Oxygen [mg/L]')


#  Biomassa na Descida
plt.subplot(4, 1, 3)
for i in range(N_downcomer + 1):
    plt.plot(t, resultados_biomassa_downcomer[i], label=f'Volume {i} (Downcomer)')
plt.xlabel('Time [h]')
plt.ylabel(' Biomass [mg/L]')

#  Produto (PHB) na Descida
plt.subplot(4, 1, 4)
for i in range(N_downcomer + 1):
    plt.plot(t, resultados_produto_downcomer[i], label=f'Volume {i} (Downcomer)')
plt.xlabel('Time [h]')
plt.ylabel(' PHB [mg/L]')
plt.show()

# Imprimir a quantidade Reactor de PHB produzido
print(f'Quantidade Reactor de PHB produzido: {quantidade_total_phb} g')
print(f'Produção: {quantidade_total_phb/(tempo)} mg/h')

Reactor_metano = (np.mean(resultados_metano_riser, axis=0)*volume_riser + np.mean(resultados_metano_gas_separator, axis=0)*volume_gas_separator + np.mean(resultados_metano_downcomer, axis=0)*volume_downcomer)/Vol_total
Reactor_oxigenio = (np.mean(resultados_oxigenio_riser, axis=0)*volume_riser + np.mean(resultados_oxigenio_gas_separator, axis=0)*volume_gas_separator + np.mean(resultados_oxigenio_downcomer, axis=0)*volume_downcomer)/Vol_total
Reactor_biomassa = (np.mean(resultados_biomassa_riser, axis=0)*volume_riser + np.mean(resultados_biomassa_gas_separator, axis=0)*volume_gas_separator + np.mean(resultados_biomassa_downcomer, axis=0)*volume_downcomer)/Vol_total
Reactor_produto = (np.mean(resultados_produto_riser, axis=0)*volume_riser + np.mean(resultados_produto_gas_separator, axis=0)*volume_gas_separator + np.mean(resultados_produto_downcomer, axis=0)*volume_downcomer)/Vol_total

# Plot the Reactor sums over time
plt.figure(figsize=(8, 12))
plt.plot(t, np.log(Reactor_metano), label='Reactor Methane Concentration  [mg/L]')
plt.plot(t, np.log(Reactor_oxigenio), label='Reactor Oxygen Concentration  [mg/L]')
plt.plot(t, np.log(Reactor_biomassa), label='Reactor Biomass Concentration  [mg/L]')
plt.plot(t, np.log(Reactor_produto), label='Reactor PHB Concentration  [mg/L]')
plt.xlabel('Time [h]',   fontsize=12)
plt.ylabel('Logarithm of the Reactor\'s components concentration [mg/L]',  fontsize=12)
plt.legend()

plt.tight_layout()
plt.show()


# Define time vector and initialize heat balance arrays
t = np.linspace(0, 40, 100)  # Adjust as needed
heat_generation_riser = np.zeros(len(t))
heat_generation_gas_separator = np.zeros(len(t))
heat_generation_downcomer = np.zeros(len(t))
heat_consumption_riser = np.zeros(len(t))
heat_consumption_gas_separator = np.zeros(len(t))
heat_consumption_downcomer = np.zeros(len(t))
net_heat_balance = np.zeros(len(t))

# Simulate hypothetical heat generation and consumption (replace with actual values)
for idx, tempo in enumerate(t):
    # Example: Heat generation due to biomass growth
    heat_generation_riser[idx] = 0.2 * idx  # Replace with actual heat generation rate
    heat_generation_gas_separator[idx] = 0.15 * idx  # Replace with actual heat generation rate
    heat_generation_downcomer[idx] = 0.1 * idx  # Replace with actual heat generation rate

    # Example: Heat consumption due to gas phase heat transfer
    heat_consumption_riser[idx] = 0.05 * idx  # Replace with actual heat consumption rate
    heat_consumption_gas_separator[idx] = 0.03 * idx  # Replace with actual heat consumption rate
    heat_consumption_downcomer[idx] = 0.02 * idx  # Replace with actual heat consumption rate

    # Calculate net heat balance for each section
    net_heat_balance[idx] = (
        heat_generation_riser[idx] - heat_consumption_riser[idx] +
        heat_generation_gas_separator[idx] - heat_consumption_gas_separator[idx] +
        heat_generation_downcomer[idx] - heat_consumption_downcomer[idx]
    )

# Plot heat generation and consumption for each section
plt.figure(figsize=(8, 12))
plt.plot(t, heat_generation_riser, 'r.', label='Heat Generation (Riser)', )
plt.plot(t, heat_consumption_riser,  'b.',label='Heat Consumption (Riser)')
plt.plot(t, heat_generation_gas_separator, 'r--', label='Heat Generation (Gas Separator)')
plt.plot(t, heat_consumption_gas_separator,  'b--',label='Heat Consumption (Gas Separator)')
plt.plot(t, heat_generation_downcomer, 'r-.',label='Heat Generation (Downcomer)')
plt.plot(t, heat_consumption_downcomer, 'b-.',label='Heat Consumption (Downcomer)')
plt.xlabel('Time [h]')
plt.ylabel('Heat [J]')
plt.title('Heat Generation and Consumption in Downcomer')
plt.legend()
plt.tight_layout()
plt.show()


# Define time vector and initialize temperature and energy arrays
t = np.linspace(0, 40, 100)  # Adjust as needed
temperature = np.zeros(len(t))
temperature[0] = 25.0  # Initial temperature at 25°C
energy_required = 0.0  # Initialize total energy required

# Simulate temperature changes based on heat generation and consumption (replace with actual values)
for idx in range(1, len(t)):
    # Example: Heat generation and consumption rates (hypothetical values)
    heat_generation_rate = (0.2 + 0.15 + 0.1) * idx  # Replace with actual heat generation rate
    heat_consumption_rate = (0.05 + 0.03 + 0.02) * idx  # Replace with actual heat consumption rate

    # Calculate the change in temperature using the heat balance equation (ΔQ = m * C * ΔT)
    delta_q = (heat_generation_rate - heat_consumption_rate)  # Change in heat
    delta_t = delta_q / (1.0 * 4.18)  # Change in temperature (assuming a specific heat capacity of 4.18 J/g°C)

    # Update the temperature and total energy required
    temperature[idx] = temperature[idx - 1] + delta_t
    energy_required += delta_q

# Print the total energy required
print(f"Total energy required to maintain constant temperature: {energy_required} Joules")