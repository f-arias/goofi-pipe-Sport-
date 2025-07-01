# Guía: Biofeedback EMG para Deportistas con `goofi-pipe`

Este documento describe cómo utilizar el software `goofi-pipe` para crear protocolos de biofeedback basados en electromiografía (EMG), orientados a la mejora del rendimiento deportivo. La flexibilidad de `goofi-pipe` lo convierte en una herramienta ideal para trabajar con atletas de cualquier disciplina.

**Dirigido a:** Entrenadores, fisioterapeutas, preparadores físicos, investigadores y atletas interesados en aplicar la tecnología de biofeedback para la optimización neuromuscular.

## ¿Por qué `goofi-pipe` para EMG?

`goofi-pipe` es una plataforma agnóstica a la señal. Su diseño se basa en el protocolo **Lab Streaming Layer (LSL)**, lo que significa que no le importa si la señal proviene de un EEG (cerebro), un ECG (corazón) o un **EMG (músculo)**. Para el software, es simplemente un flujo de datos que puede ser procesado en tiempo real.

Sus ventajas clave son:
*   **Modularidad:** Permite construir un flujo de procesamiento visualmente, conectando nodos como si fueran piezas de LEGO.
*   **Flexibilidad:** Puedes crear desde un simple medidor de activación hasta un complejo sistema de feedback auditivo y visual.
*   **Extensibilidad:** Los datos procesados pueden enviarse a otras aplicaciones (motores de videojuegos, software de análisis, etc.) a través de OSC.

---

## Flujo de Trabajo Típico: De la Señal al Feedback

El siguiente diagrama y descripción detallan un flujo de trabajo estándar para un protocolo de biofeedback EMG.

```
[Dispositivo EMG con LSL] -> [goofi-pipe] -> [Feedback Visual/Auditivo]
```

### Paso 1: Captura de la Señal EMG

*   **Nodo principal:** `` `LslClient` ``
*   **Acción:** Se configura este nodo para que reciba el flujo de datos que tu dispositivo EMG está transmitiendo a través de la red local vía LSL. En este punto, `goofi-pipe` está recibiendo la señal eléctrica cruda del músculo.

### Paso 2: Procesamiento para Extraer la Activación Muscular

La señal EMG cruda necesita ser procesada para convertirla en una medida útil de "intensidad" o "nivel de contracción". La cadena de nodos de procesamiento sería:

1.  **Filtro Pasa Banda (`Band-pass Filter`):**
    *   **Propósito:** Limpiar la señal cruda. Elimina el ruido eléctrico de baja frecuencia (artefactos de movimiento) y de alta frecuencia (ruido de la red eléctrica). Esto aísla la señal muscular real.

2.  **Rectificación (`Math` -> `abs`):**
    *   **Propósito:** La señal EMG contiene valores tanto positivos como negativos que indican actividad. Se utiliza un nodo matemático para aplicar el **valor absoluto (`abs`)**, convirtiendo todos los valores a positivos.

3.  **Suavizado de la Envolvente (`Low-pass Filter` o `Moving Average`):**
    *   **Propósito:** Este es el paso crucial. La señal rectificada sigue siendo muy errática. Al aplicar un filtro de paso bajo o una media móvil, se obtiene una línea suave que representa fielmente el **nivel de activación general del músculo**. Esta señal suavizada es la que se utilizará para el feedback.

### Paso 3: Biofeedback en Tiempo Real

Con una señal limpia que representa la contracción muscular, se envía a un nodo de salida para proporcionar feedback al atleta:

*   **Feedback Visual (`Plot`):**
    *   Muestra la activación como una barra, un medidor o una línea que sube y baja. Es un feedback directo e intuitivo para que el atleta **vea** su nivel de contracción.

*   **Feedback Auditivo (`Biotuner` o `Sonification`):**
    *   Convierte el nivel de activación en sonido. Por ejemplo, el tono o el volumen de una nota aumenta con la contracción. Esto libera al atleta de mirar una pantalla, permitiéndole concentrarse en la sensación corporal (propiocepción).

---

## Aplicaciones Prácticas para el Rendimiento Deportivo

Esta configuración se puede adaptar para abordar objetivos específicos en cualquier disciplina:

*   **Entrenamiento de Activación Muscular:**
    *   **Objetivo:** Enseñar a un atleta a reclutar un músculo específico que está infrautilizado (ej. glúteo medio en corredores, oblicuos en lanzadores).
    *   **Feedback:** El atleta intenta que la barra visual o el tono auditivo suban al realizar un movimiento concreto.

*   **Entrenamiento de Relajación y Control:**
    *   **Objetivo:** Ayudar a reducir la tensión parásita en músculos que se sobrecargan y limitan el rendimiento (ej. trapecio superior en arqueros, golfistas o tenistas).
    *   **Feedback:** El atleta intenta mantener la señal visual o auditiva lo más baja posible durante la preparación y ejecución del gesto técnico.

*   **Entrenamiento de Simetría Bilateral:**
    *   **Objetivo:** Corregir desequilibrios de fuerza o activación entre el lado izquierdo y derecho del cuerpo.
    *   **Requisitos:** Un EMG con al menos dos canales.
    *   **Feedback:** Se muestran dos barras, una para cada lado. El objetivo es que ambas suban de forma simétrica durante ejercicios como sentadillas, remos o peso muerto.

*   **Análisis de Fatiga Muscular (Avanzado):**
    *   **Objetivo:** Monitorizar los cambios en la señal EMG durante un esfuerzo prolongado para identificar objetivamente la aparición de la fatiga.
    *   **Herramienta:** Se utiliza el nodo `PSD` para analizar cómo cambia el espectro de frecuencia de la señal, un indicador fisiológico de la fatiga.

## Requisitos Técnicos

Para implementar este sistema, necesitas:
1.  **Hardware:** Un dispositivo EMG (ej. OpenBCI, g.tec, etc.) que sea capaz de transmitir sus datos mediante el protocolo LSL.
2.  **Software:**
    *   Una instalación funcional de `goofi-pipe`.
    *   El software o la aplicación puente que toma los datos de tu dispositivo EMG y los retransmite a la red local vía LSL.

**Nota Importante:** Este software se proporciona bajo la licencia MIT. Se entrega "tal cual", sin ninguna garantía. No es un dispositivo médico certificado y cualquier uso en un contexto de salud o rendimiento es bajo la total responsabilidad del usuario.
