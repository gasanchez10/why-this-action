import nbformat as nbf
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): C.append(nbf.v4.new_code_cell(s.strip()))

md(r'''
# Why This Action? — 05 · El paquete completo en un solo recorrido

Los cuadernos 01 a 04 construyeron las piezas sobre dos ejemplos. Este cuaderno usa el paquete general `wta/`, que sirve para cualquier grafo, con mecanismos lineales o no lineales, y que es el que corren los experimentos del artículo (`experiments/run.py`).

| § | Qué muestra | Dónde está en el artículo | Módulo |
|---|---|---|---|
| 1 | El problema: grafo, datos, caja negra, versiones del modelo causal | §2–§3, Definiciones 1–5 | `wta/data.py`, `wta/scm.py` |
| 2 | Buscar el plan: Algoritmo 1 con su registro de evaluaciones | §5, Algoritmo 1, Proposición 12 | `wta/search.py` |
| 3 | Preguntas sobre el plan | §4, Definición 9 | `wta/explain.py` |
| 4 | Créditos por camino y fidelidad | §4, Definición 10, Lema 11 | `wta/explain.py` |
| 5 | La trayectoria de búsqueda como explicación | §6, Proposición 13 | `wta/explain.py` |
| 6 | Extensión: las tres capas de error | No está en el borrador | `wta/extensions.py` |
| 7 | Los experimentos | §7 | `experiments/` |

Todo lo que se imprime es simulación con una semilla. Las tablas con varias semillas están en `results/SUMMARY.md`.
''')
md("## 1 · El problema")
code(r'''
import sys, os; sys.path.insert(0, os.path.abspath(".."))
import numpy as np
from wta.data import CASES, BlackBox
from wta.scm import Ensemble
from wta.search import solve, reference, robust
from wta import explain as ex, extensions as xt

case = CASES["perioperative"](seed=1)             # cambiar por "loan7", "dag10", "synth3-anm", "german", ...
sp = case.spec
print("Variables:", sp.names)
print("Flechas:", [(sp.names[k], sp.names[j]) for k, j in sp.edges])
print("Se puede cambiar:", {v: (float(lo), float(hi)) for v, lo, hi in zip(sp.actionable, sp.lo, sp.hi)})
print("Se puede bloquear:", [(sp.names[k], sp.names[j]) for k, j in sp.blk])

ens = Ensemble(sp, case.X, S=30, kind=case.kind, seed=1)     # versión 0 = estimación puntual; 1..29 = re-ajustes por bootstrap
x = case.individuals(1, np.random.default_rng(5))[0]
print("\nPaciente:", dict(zip(sp.names, x.round(2))), "| margen de la caja negra:", round(float(case.blackbox().margin(x)[0]), 3), "(negativo = no apto)")
''')
md(r'''
## 2 · Buscar el plan (Algoritmo 1)

**Qué hace.** Busca el plan más barato que cruza el umbral en al menos el 90 % de las versiones. El proceso gaussiano aprende la caja negra sobre sus **entradas**, no sobre los planes; cada paso evalúa la caja negra una sola vez, en la versión que decide la validez; un plan solo pasa a ser el mejor cuando se evaluó la caja negra real en todas las versiones (certificación); al final se intenta quitar cada cambio y cada bloqueo (poda).

**Qué mirar.** El número de consultas por propósito, y que el plan queda cerca del óptimo de referencia (que usa la caja negra sin límite).
''')
code(r'''
EPS = 0.10
bb = case.blackbox(); res = solve(sp, ens, x, bb, EPS, budget=50, scale_X=case.X, seed=1)
print("Plan:", sp.describe(res.action), "| costo", round(res.cost, 3))
print("Consultas por propósito:", res.queries, "| total", bb.n)
free = lambda X: BlackBox(case.proba).margin(X)                 # evaluaciones que no se cuentan: solo para medir
ref, rc = reference(sp, ens, x, free, EPS)
print("Óptimo de referencia:", sp.describe(ref), "| costo", round(rc, 3), f"| brecha {100 * (res.cost / rc - 1):.1f} %")
print("¿Funciona en el modelo causal verdadero?", bool(free(case.true.counterfactual(x, res.action)[0])[0] >= 0))
print("\nPrimeras y últimas líneas del registro:")
for r in res.trace[:3] + res.trace[-3:]: print(f"  paso {r['step']:>3} | {r['purpose']:<15} | {sp.describe(r['action']):<45} | margen {r['margin']:+.3f} | costo {r['cost']:.3f}")
''')
md(r'''
## 3 · Preguntas sobre el plan

**Qué hace.** Responde, con las mismas versiones del modelo causal: ¿y si el cambio se logra solo en parte? ¿y si una relación causal no se cumple? ¿y si el bloqueo no se sostiene? ¿qué plan habría sin tocar cierta variable?

**Qué mirar.** El modo con proceso gaussiano responde casi sin consultas cuando la respuesta es clara, y consulta la caja negra solo cerca del umbral.
''')
code(r'''
a = res.action
print("Plan tal cual:            ", {k: v for k, v in ex.answer(ens, x, a, case.blackbox(), EPS).items() if k in ("valid", "p_success", "ci", "queries")})
print("Solo el 80 % de cada cambio:", {k: v for k, v in ex.answer(ens, x, a, case.blackbox(), EPS, gamma=0.8).items() if k in ("valid", "p_success", "queries")})
print("  igual, modo GP:           ", {k: v for k, v in ex.answer(ens, x, a, case.blackbox(), EPS, gamma=0.8, surrogate=res.surrogate).items() if k in ("valid", "p_success", "queries")})
print("\nFracción mínima de cada cambio que mantiene el plan válido:", ex.smallest_fractions(ens, x, a, case.blackbox(), EPS))
print("Relaciones cuya falla sola tumba el plan (¿lo tumba?, éxito):", ex.single_points_of_failure(ens, x, a, case.blackbox(), EPS))
print("Si el bloqueo no se sostiene (¿sigue válido?, éxito):", ex.block_reliance(ens, x, a, case.blackbox(), EPS))
alt = ex.alternative(sp, ens, x, case.blackbox(), EPS, forbid_vars=[sp.actionable[int(np.argmax(np.abs(a[:sp.nA])))]], budget=40, scale_X=case.X, seed=1)
print("\nAlternativa sin tocar la variable que más cambia:", sp.describe(alt.action), "| costo", round(alt.cost, 3))
''')
md(r'''
## 4 · ¿Por qué esta acción? Créditos por camino

**Qué hace.** Reparte la ganancia del plan entre los caminos causales por los que actúa (valores de Shapley del juego de caminos) y mide cuánto interactúan los caminos (fidelidad: 1 = no interactúan; la definición exacta hay que cotejarla con la ecuación 11 del borrador).

**Qué mirar.** Los créditos suman exactamente la ganancia del plan (Lema 11).
''')
code(r'''
g = ex.explanation_game(ens, x, a, case.blackbox())
for lab, c in sorted(zip(g["labels"], g["credits"]), key=lambda t: -abs(t[1])): print(f"  {c:+.3f}  {lab}")
gain = free(ens.counterfactual(x, a)[0, 0])[0] - free(x)[0]
print(f"Suma de créditos {g['credits'].sum():+.4f} = ganancia del plan {gain:+.4f} | fidelidad {g['fidelity']:.3f} (variante con valores absolutos: {g['fidelity_abs']:.3f})")
print("Por relación causal:", {k: round(v, 3) for k, v in ex.aggregate(ens, g)[0].items()})
''')
md(r'''
## 5 · La trayectoria de búsqueda como explicación

**Qué hace.** Usa el registro de la búsqueda: cómo difieren los mejores planes sucesivos, por qué camino se queda corto el plan más barato que falló, y qué aporta la propagación causal frente a hacer los mismos cambios sin grafo.
''')
code(r'''
tc = ex.trajectory_contrasts(ens, x, res, case.blackbox())
for j in tc["jumps"]:
    print(f"De [{j['frm']}] a [{j['to']}]: ahorra {j['saved']:.3f} de costo, cambia el margen en {j['game']['total']:+.3f}")
if tc["near_miss"]:
    nm = tc["near_miss"]; print(f"\nPlan más barato que falló: [{nm['action']}] costo {nm['cost']:.3f}, margen robusto {nm['robust_margin']:+.3f}. Lo que le falta frente al plan final:")
    for lab, c in sorted(zip(nm["game"]["labels"], nm["game"]["credits"]), key=lambda t: -abs(t[1]))[:4]: print(f"  {c:+.3f}  {lab}")
gg = tc["graph"]["game"]; print(f"\nLo que aporta la propagación causal al margen: {gg['total']:+.3f} (repartido en {len(gg['paths'])} caminos)")
print("\n" + ex.ficha(case, ens, x, a, case.blackbox(), EPS))
''')
md(r'''
## 6 · Extensión: las tres capas de error

**Qué hace.**
- **Capa 1, el grafo.** Re-estima el grafo en remuestreos y arma versiones del modelo causal que no comparten grafo.
- **Capa 2, los mecanismos.** Además del bootstrap, calcula para cada mecanismo que el plan atraviesa un intervalo conformal localizado en el punto al que el plan lleva a sus padres.
- **Capa 3, la recomendación.** Probabilidad, dada por el experto, de que un bloqueo no se sostenga.

**Literatura y límites.** Ver `THEORY.md`: qué está demostrado (suma de errores entre capas; caso lineal con orden conocido) y qué no (cobertura del bootstrap de grafos; conformal en un punto).

**Qué mirar.** La probabilidad de éxito baja al añadir capas; el plan al borde de la validez por bootstrap no pasa la prueba conformal; el plan buscado con la caja conformal es más caro, recurre al bloqueo o no existe. La razón está en `THEORY.md` §3: el intervalo conformal incluye el ruido de un paciente nuevo, que para este paciente se cancela.
''')
code(r'''
G = xt.bootstrap_graphs(case.X, B=60, order=sp.order, seed=1)
print("Grafos distintos en 60 remuestreos:", len(G), "| peso del más frecuente:", G[0][1])
mg = xt.graph_ensemble(sp, case.X, G, S=40, kind=case.kind, seed=1)
print("Probabilidad de éxito del plan por capas:", xt.layered_success(ens, mg, x, a, free, block_fail=0.03))
bb = case.blackbox(); res_g = solve(sp, mg, x, bb, EPS, budget=50, scale_X=case.X, seed=1)
print("Plan que cubre también la incertidumbre del grafo:", sp.describe(res_g.action), "| costo", round(res_g.cost, 3))

half = len(case.X) // 2; fit = Ensemble(sp, case.X[:half], S=1, kind=case.kind, seed=1)
cp = xt.conformal_plan(fit, case.X[half:], x, a, free, alpha=0.10)
print("\nPrueba conformal del plan de la sección 2:", {k: cp[k] for k in ("L", "half_width", "n_eff", "worst_margin")})
box = xt.ConformalBoxSCM(fit, case.X[half:], alpha=0.10, seed=1); res_c = solve(sp, box, x, case.blackbox(), 0.0, budget=50, scale_X=case.X, seed=1)
print("Plan buscado con la caja conformal:", sp.describe(res_c.action), "| costo", round(res_c.cost, 3))
''')
md(r'''
## 7 · Los experimentos

```
python -m experiments.run all --mode quick     # tamaños pequeños, para comprobar que todo corre
python -m experiments.run all --mode full      # 10 semillas de datos y 5 individuos por caso (horas)
python -m experiments.run C --cases loan7 --seeds 5 --n 4
python -m experiments.summarize                # escribe results/SUMMARY.md
```

| Experimento | Qué mide |
|---|---|
| A | Si las respuestas a las preguntas coinciden con lo que pasa en el modelo causal verdadero; modo exacto contra modo GP; créditos |
| B | Cuándo el bloqueo abarata el plan o lo hace posible |
| C | Buscador: brecha frente al óptimo de referencia, validez real y consultas, con ablaciones y líneas de base |
| D | La ficha para los clínicos |
| E | Estabilidad de costos, créditos y contrastes entre semillas de búsqueda |
| X | Las tres capas de error |

**Lo que NO está implementado como en los artículos originales** (ver `README.md`): Shapley Flow, DAG-SHAP, causal Shapley, CF-SHAP, BayCon, BO causal con restricciones y la variante por gradiente de Karimi et al. Donde hay un sustituto, lleva la palabra *stand-in* en el nombre.
''')
nb = nbf.v4.new_notebook(); nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
nb.cells = C; nbf.write(nb, "05_paquete_completo.ipynb"); print("written", len(C))
