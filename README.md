# Why This Action? — código del artículo

Código de *Why This Action? DAG-Driven, Causally Aware Counterfactual Explanations for Black-Box Models* (borrador `main_v2`), más la extensión de las tres capas de error (grafo, mecanismos, recomendación).

**Todo lo que produce este código es simulación.** Los números de `results/SUMMARY.md` salen de una corrida corta (modo `quick`) y sirven para comprobar que todo corre, no para el artículo. Para el artículo hay que correr el modo `full` y revisar las advertencias de abajo.

## Cómo se usa

```
pip install numpy scipy scikit-learn pandas tabulate nbformat nbclient matplotlib
python tests/test_propositions.py              # 9 comprobaciones numéricas de las proposiciones (1 minuto)
python -m experiments.run all --mode quick     # todos los experimentos, tamaños pequeños (alrededor de 1 hora en 2 núcleos)
python -m experiments.run all --mode full      # 10 semillas de datos y 5 individuos por caso (muchas horas; conviene lanzar un caso por proceso)
python -m experiments.run C --cases loan7 --seeds 5 --n 4
python -m experiments.summarize                # escribe results/SUMMARY.md
```

Los resultados se **añaden** a los CSV de `results/`. Para empezar de cero, borre esa carpeta o use otra con la variable `WTA_RESULTS`.

## Qué hay en cada archivo

| Archivo | Contenido | Parte del artículo |
|---|---|---|
| `wta/scm.py` | Grafo y espacio de acciones (`Spec`); versiones del modelo causal por bootstrap, lineales o no lineales (`Ensemble`); modelo verdadero (`TrueSCM`); operador contrafactual con cambios y bloqueos; estados restringidos a caminos; contribuciones por camino del caso lineal | Definiciones 1, 3, 4; Proposiciones 7 y 8; Apéndice A |
| `wta/search.py` | Validez robusta; Algoritmo 1 (proceso gaussiano sobre las entradas de la caja negra, mejora factible esperada, certificación, poda, registro); óptimo de referencia; líneas de base y ablaciones | Definiciones 5 y 6; §5; Proposición 12 |
| `wta/explain.py` | Preguntas sobre el plan en modo exacto y modo GP; juego de caminos, créditos de Shapley, fidelidad; contrastes de la trayectoria; ficha para clínicos | Definiciones 9 y 10; Lema 11; §6; Proposición 13; Experimento D |
| `wta/extensions.py` | Bootstrap de grafos (con orden dado o sin él); modelo causal con versiones de distinto grafo; probabilidad de éxito por capas; conformal localizado dentro del plan; caja conformal; GP-SCM de Karimi et al. 2020 reimplementado | Extensión, no está en el borrador |
| `wta/data.py` | Los casos: perioperatorio, tres variables de Karimi et al. (lineal y no lineal), grafos aleatorios de 10 a 30 nodos, préstamos de 7 variables, German Credit, Adult, Sachs | §7 |
| `experiments/run.py` | Experimentos A, B, C, D, E y X (extensión) | §7 |
| `experiments/summarize.py` | Tablas con intervalos de confianza | |
| `tests/test_propositions.py` | Proposiciones 7, 8, 12, 13, Lema 11, operador contra la fórmula cerrada | |
| `THEORY.md` | Enunciados y pruebas de la extensión: suma de errores entre capas, garantía por caja y sus condiciones, cota por número de mecanismos bajo intervención, validez marginal y condicional | Extensión |
| `notebooks/` | Cuadernos 01 a 04 (los prototipos, con `wta_core.py` y `wta_loan.py`) y 05 (recorrido por el paquete) | |
| `data_raw/` | German Credit, Adult y Sachs, descargados de repositorios públicos | |

## Lo que NO es lo que dice el artículo (leer antes de escribir resultados)

**Líneas de base que no están implementadas como en sus artículos.** El borrador nombra Shapley Flow, DAG-SHAP, causal Shapley, CF-SHAP, BayCon, BO causal con restricciones y Karimi et al. 2020 (gradiente y malla). Aquí hay:

| En el borrador | En el código | Estado |
|---|---|---|
| Karimi et al. 2020, malla | `GPSCM` + `grid_search` | Reimplementado por nosotros desde el artículo. No es su código. Solo corre en casos con hasta 4 variables accionables |
| Karimi et al. 2020, gradiente | — | **Falta.** Necesita un clasificador diferenciable |
| BO causal con restricciones, sin abducción | `solve(..., kw=dict(abduct=False))` | *Stand-in*: nuestro mismo buscador, pero el ruido del individuo se reemplaza por un residuo al azar. No es el método publicado |
| BayCon | `solve(..., propagate=False)` | *Stand-in* no causal: mismos cambios sin propagarlos. No es BayCon |
| Shapley Flow, DAG-SHAP, causal Shapley, CF-SHAP | `action_level_shapley` | *Stand-in*: Shapley sobre los componentes del plan. **Ninguno de los cuatro está implementado** |
| Búsqueda al azar | `random_search` | Implementada |

No se puede reportar ningún número con el nombre de un método publicado mientras lleve la palabra *stand-in*.

**Decisiones nuestras que hay que cotejar.**
- La regla de etiquetas de los casos de tres variables (`synth3`): las ecuaciones del modelo causal salen del código público de Karimi et al.; la regla de etiquetas es nuestra.
- El grafo de German Credit (sexo → monto, edad → monto, monto → duración) y el de Adult están escritos de memoria. **A verificar** contra Karimi et al. 2021 y Nabi y Shpitser / Chiappa.
- En Sachs, la tarea de clasificación (Akt por encima de su mediana) la pusimos nosotros para tener una caja negra.
- La fidelidad F: se calcula como la leímos de la ecuación 11 (sin valores absolutos) y también con valores absolutos. Cotejar con la fuente.
- El margen es logit(h(x)) − logit(τ), no h(x) − τ. El signo es el mismo.
- El caso perioperatorio usa una red neuronal como caja negra (`model="gbm"` da *gradient boosting*, como dice el borrador).
- El presupuesto del Algoritmo 1 cuenta las evaluaciones de búsqueda. Las de certificación y poda se cuentan aparte y aparecen en la columna de consultas.
- `polish=True` (encoger cada cambio mientras el plan siga siendo válido) es un añadido nuestro; no está en el Algoritmo 1.

**Datos reales.** Sin modelo causal verdadero no existe "validez real". Los experimentos usan como sustituto un re-ajuste con datos apartados y lo dicen en la columna `truth`.

**Garantías.** Lo que está demostrado y lo que no está en `THEORY.md`. En corto: el bootstrap de grafos no tiene garantía; la caja la tiene solo con mecanismos lineales, orden conocido y margen afín; el conformal localizado es aproximado y, además, demasiado ancho para un individuo porque incluye el ruido.

## Pendiente

1. Correr el modo `full` y decidir qué tablas van al artículo.
2. Las líneas de base de atribución (Shapley Flow y las demás) y la variante por gradiente de Karimi et al.
3. Integrar `recourse_bo` (el código propio del autor), que no estaba disponible al escribir este paquete.
4. Datos perioperatorios reales y la evaluación con anestesiólogos (Experimento D); aquí solo se genera la ficha.
5. Descubrimiento causal para muchas variables con un método establecido; `bootstrap_graphs` sin orden usa una búsqueda golosa aproximada.
6. Un procedimiento conformal para la diferencia contrafactual del individuo (ver `THEORY.md` §3).
