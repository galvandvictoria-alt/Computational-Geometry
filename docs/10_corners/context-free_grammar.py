import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math

# =============================================
# 1. CARGAR IMAGEN Y EXTRAER CONTORNO
# =============================================

def cargar_imagen_binaria(ruta):
    img = Image.open(ruta).convert('L')
    arr = np.array(img)
    return np.where(arr > 127, 255, 0).astype(np.uint8)


def extraer_contorno_ordenado(binary_img):
    """
    Extrae el contorno como lista ordenada de puntos (x, y)
    comenzando desde el píxel más arriba-izquierda.
    """
    filas, cols = binary_img.shape

    # Encontrar punto de inicio: primer píxel de objeto con fondo arriba o izquierda
    start = None
    for f in range(filas):
        for c in range(cols):
            if binary_img[f, c] == 255:
                if f == 0 or binary_img[f-1, c] == 0:
                    start = (c, f)  # (x, y)
                    break
        if start:
            break

    if start is None:
        return []

    # Trazar contorno con F8 para obtener puntos ordenados
    # Direcciones F8: 0=der, 1=der-abajo, 2=abajo, 3=izq-abajo,
    #                 4=izq, 5=izq-arriba, 6=arriba, 7=der-arriba
    moves = {0:(1,0), 1:(1,1), 2:(0,1), 3:(-1,1),
             4:(-1,0), 5:(-1,-1), 6:(0,-1), 7:(1,-1)}

    x, y = start
    contorno = [(x, y)]
    dir_actual = 0

    for _ in range(100000):
        # Buscar siguiente píxel de contorno
        # Empezar buscando desde dirección opuesta+1 (sentido antihorario)
        dir_busqueda = (dir_actual + 5) % 8
        encontrado = False

        for _ in range(8):
            dx, dy = moves[dir_busqueda]
            nx, ny = x + dx, y + dy
            if 0 <= ny < filas and 0 <= nx < cols:
                if binary_img[ny, nx] == 255:
                    dir_actual = dir_busqueda
                    x, y = nx, ny
                    encontrado = True
                    break
            dir_busqueda = (dir_busqueda + 1) % 8

        if not encontrado:
            break

        if (x, y) == start and len(contorno) > 2:
            break

        contorno.append((x, y))

    return contorno


# =============================================
# 2. CÓDIGO AF8 DEL CONTORNO
# =============================================

def contorno_a_f8(contorno):
    """Convierte lista de puntos a código F8."""
    dir_map = {(1,0):0, (1,1):1, (0,1):2, (-1,1):3,
               (-1,0):4, (-1,-1):5, (0,-1):6, (1,-1):7}
    f8 = []
    n = len(contorno)
    for i in range(n):
        cx, cy = contorno[i]
        nx, ny = contorno[(i+1) % n]
        dx, dy = nx - cx, ny - cy
        if (dx, dy) in dir_map:
            f8.append(dir_map[(dx, dy)])
    return f8


def f8_a_af8(f8):
    """Convierte F8 absoluto a AF8 relativo."""
    if not f8:
        return []
    af8 = []
    n = len(f8)
    for i in range(n):
        prev = f8[i-1]
        curr = f8[i]
        af8.append((curr - prev) % 8)
    return af8


# =============================================
# 3. CÁLCULO DE ISE ENTRE DOS PUNTOS DEL CONTORNO
# =============================================

def calcular_ise_segmento(contorno, idx_k, idx_k1):
    """
    Calcula ISE entre break points P_k y P_{k+1}.
    Usa la fórmula del paper (Definición 2.5 y 2.6):
    d²(pi, Pk->Pk+1) = ((xi-xk)(yk+1-yk) - (yi-yk)(xk+1-xk))²
                       / ((xi-xk+1)² + (yi-yk+1)²)
    sumado para todos los puntos del contorno entre k y k+1.
    """
    n = len(contorno)
    xk, yk   = contorno[idx_k]
    xk1, yk1 = contorno[idx_k1]

    # Distancia euclidiana entre los dos break points (denominador del ISE compartido)
    d = math.sqrt((xk1 - xk)**2 + (yk1 - yk)**2)
    if d == 0:
        return 0.0

    # Recorrer puntos del contorno entre k y k+1
    ise = 0.0
    idx = (idx_k + 1) % n
    while idx != idx_k1:
        xi, yi = contorno[idx]
        # Distancia del punto al segmento de línea
        num = ((xi - xk) * (yk1 - yk) - (yi - yk) * (xk1 - xk)) ** 2
        den = (xk1 - xk)**2 + (yk1 - yk)**2
        if den > 0:
            ise += num / den
        idx = (idx + 1) % n

    return ise


def longitud_arco(contorno, idx_k, idx_k1):
    """Longitud del arco (número de puntos) entre idx_k e idx_k1."""
    n = len(contorno)
    s = 0
    idx = idx_k
    while idx != idx_k1:
        idx = (idx + 1) % n
        s += 1
    return s


# =============================================
# 4. DETECCIÓN DE BREAK POINTS (AF8 + GRAMÁTICA)
# =============================================

def detectar_break_points(contorno, af8, p_max, q_max, r_max):
    """
    Detecta break points usando la gramática libre de contexto del paper.
    L = {x a^p (bh a^q)^r, x a^p (hb a^q)^r | x ∈ {a,b,c,d,e,f,g,h}}

    En AF8: a=0, b=1, c=2, d=3, e=4, f=5, g=6, h=7
    Un break point ocurre donde cambia el patrón de segmento recto.
    """
    n = len(af8)
    if n == 0:
        return list(range(len(contorno)))

    break_points = []
    i = 0

    while i < n:
        # El símbolo actual marca un posible break point
        break_points.append(i)

        # Intentar consumir el mayor segmento recto posible
        # Patrón: a^p (patron_diagonal a^q)^r
        a_sym = af8[i]  # El símbolo dominante en esta dirección

        # Contar a's iniciales (p)
        p = 0
        j = (i + 1) % n
        while p < p_max and af8[j] == a_sym:
            p += 1
            j = (j + 1) % n
            if j == i:
                break

        # Intentar patrón (bh a^q)^r o (hb a^q)^r
        b_sym = (a_sym + 1) % 8
        h_sym = (a_sym + 7) % 8

        r = 0
        while r < r_max:
            # Verificar patrón bh o hb
            s1, s2 = af8[j % n], af8[(j+1) % n]
            if (s1 == b_sym and s2 == h_sym) or (s1 == h_sym and s2 == b_sym):
                j = (j + 2) % n
                # Contar a^q
                q = 0
                while q < q_max and af8[j % n] == a_sym:
                    q += 1
                    j = (j + 1) % n
                    if j % n == i:
                        break
                r += 1
            else:
                break

        # Avanzar al siguiente break point
        if j == i or (r == 0 and p == 0):
            i = (i + 1) % n
        else:
            i = j % n

        if i == 0 and len(break_points) > 1:
            break

    # Asegurarse que los índices sean únicos y ordenados
    break_points = sorted(set(break_points))
    return break_points


def detectar_break_points_simple(contorno):
    """
    Detección simplificada de break points: puntos donde cambia
    significativamente la dirección del contorno (cambios en F8).
    Más robusta para imágenes reales.
    """
    n = len(contorno)
    if n < 3:
        return list(range(n))

    f8 = contorno_a_f8(contorno)
    af8 = f8_a_af8(f8)

    break_points = []

    for i in range(n):
        # Un break point ocurre donde el cambio relativo no es 0
        # (es decir, donde hay cambio de dirección)
        if af8[i] != 0:
            break_points.append(i)

    # Si hay muy pocos, usar todos
    if len(break_points) < 4:
        return list(range(0, n, max(1, n//20)))

    return break_points


# =============================================
# 5. ELIMINACIÓN DE BREAK POINTS
# =============================================

def eliminar_break_points(contorno, break_points, T):
    """
    Elimina break points innecesarios.
    
    List1 = [ISE(k,k+1), ISE(k+1,k+2), ...]
    List2 = [ISE(k,k+2), ISE(k+1,k+3), ...] (alternados)
    
    Elimina P_{k+1} si ISE_total - ISE(k,k+1) - ISE(k+1,k+2) + ISE(k,k+2) <= T
    """
    bps = list(break_points)
    m = len(bps)

    if m <= 3:
        return bps

    # Calcular List1: ISE entre break points consecutivos
    def get_ise(i, j):
        return calcular_ise_segmento(contorno, bps[i % m], bps[j % m])

    # Recalcular todo cuando cambia la lista
    improved = True
    while improved and len(bps) > 3:
        improved = False
        m = len(bps)

        # List1
        list1 = [calcular_ise_segmento(contorno, bps[i], bps[(i+1) % m])
                 for i in range(m)]
        ISE_total = sum(list1)

        if ISE_total > T:
            break

        # List2: ISE saltando un punto
        list2 = []
        for i in range(m):
            ise_skip = calcular_ise_segmento(contorno, bps[i], bps[(i+2) % m])
            list2.append((ise_skip, i))

        # Encontrar ISEMin en List2
        list2.sort(key=lambda x: x[0])
        ise_min, idx_k = list2[0]

        # Calcular nuevo ISE total si eliminamos bps[idx_k + 1]
        idx_k1 = (idx_k + 1) % m
        idx_k2 = (idx_k + 2) % m

        ise_k_k1  = list1[idx_k]
        ise_k1_k2 = list1[idx_k1]

        new_ISE_total = ISE_total - ise_k_k1 - ise_k1_k2 + ise_min

        if new_ISE_total <= T:
            # Eliminar bps[idx_k1]
            bps.pop(idx_k1)
            improved = True

    return bps


# =============================================
# 6. REORDENAMIENTO 
# =============================================

def calcular_ise_entre_puntos(contorno, pk, pk1):
    """ISE entre dos puntos del contorno dados directamente."""
    n = len(contorno)
    # Encontrar índices
    try:
        idx_k  = contorno.index(pk)
        idx_k1 = contorno.index(pk1)
    except ValueError:
        return float('inf')
    return calcular_ise_segmento(contorno, idx_k, idx_k1)


def vecinos_cercanos(contorno, idx, radio=2):
    """Devuelve los índices de los vecinos dentro de radio dado."""
    n = len(contorno)
    vecinos = []
    for d in range(-radio, radio+1):
        v_idx = (idx + d) % n
        vecinos.append(v_idx)
    return vecinos


def reordenar_break_points(contorno, break_points):
    """
    Reordenamiento de break points para minimizar ISE total.
    Versión simplificada del algoritmo de Dijkstra del paper.
    Prueba pequeños desplazamientos de cada break point y acepta
    si reduce el ISE total.
    """
    n = len(contorno)
    bps = list(break_points)
    m = len(bps)

    if m <= 3:
        return bps

    def ise_total(bp_list):
        total = 0
        for i in range(len(bp_list)):
            total += calcular_ise_segmento(
                contorno, bp_list[i], bp_list[(i+1) % len(bp_list)])
        return total

    mejor_ise = ise_total(bps)
    mejorado = True

    while mejorado:
        mejorado = False
        for i in range(m):
            vecinos = vecinos_cercanos(contorno, bps[i], radio=2)
            for v in vecinos:
                if v == bps[i]:
                    continue
                # Verificar que no colisione con otros break points
                if v in bps:
                    continue
                nuevos_bps = list(bps)
                nuevos_bps[i] = v
                nuevo_ise = ise_total(nuevos_bps)
                if nuevo_ise < mejor_ise:
                    mejor_ise = nuevo_ise
                    bps = nuevos_bps
                    mejorado = True
                    break

    return bps


# =============================================
# 7. ALGORITMO PRINCIPAL
# =============================================

def aproximacion_poligonal(contorno, T):
    """
    1. Detección de break points
    2. Eliminación de break points
    3. Reordenamiento
    Repite hasta convergencia.
    """
    n = len(contorno)
    print(f"  Contorno: {n} puntos")

    # Paso 1: Detección inicial de break points
    bps = detectar_break_points_simple(contorno)
    print(f"  Break points iniciales: {len(bps)}")

    # Calcular ISE inicial
    def ise_total(bp_list):
        total = 0
        m = len(bp_list)
        for i in range(m):
            total += calcular_ise_segmento(
                contorno, bp_list[i], bp_list[(i+1) % m])
        return total

    ise = ise_total(bps)
    print(f"  ISE inicial: {ise:.3f}  (T={T})")

    # Iterar eliminación + reordenamiento hasta convergencia
    list_anterior = None
    iteracion = 0
    max_iter = 20

    while iteracion < max_iter:
        iteracion += 1

        # Paso 2: Eliminación
        bps = eliminar_break_points(contorno, bps, T)

        # Paso 3: Reordenamiento
        bps = reordenar_break_points(contorno, bps)

        # Verificar convergencia
        ise_nuevo = ise_total(bps)
        print(f"  Iter {iteracion}: DP={len(bps)}, ISE={ise_nuevo:.3f}")

        if bps == list_anterior:
            print("  Convergencia alcanzada.")
            break
        list_anterior = list(bps)

    return bps


# =============================================
# 8. VISUALIZACIÓN
# =============================================

def visualizar_resultado(binary_img, contorno, break_points_indices, T, ruta):
    """
    Muestra:
    - Imagen binaria original
    - Contorno con break points marcados
    - Polígono aproximado (como en Figura 9 del paper)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Aproximación Poligonal  |  T={T}  |  DP={len(break_points_indices)}',
                 fontsize=13)

    # --- Imagen original ---
    axes[0].imshow(binary_img, cmap='gray')
    axes[0].set_title('Imagen binaria')
    axes[0].axis('off')

    # --- Contorno + break points ---
    axes[1].imshow(binary_img, cmap='gray')
    cx = [p[0] for p in contorno]
    cy = [p[1] for p in contorno]
    axes[1].plot(cx, cy, 'c-', linewidth=0.8, alpha=0.6, label='Contorno')

    bpx = [contorno[i][0] for i in break_points_indices]
    bpy = [contorno[i][1] for i in break_points_indices]
    axes[1].plot(bpx, bpy, 'ro', markersize=4, label=f'Break points ({len(break_points_indices)})')
    axes[1].set_title(f'Contorno + Break points')
    axes[1].legend(fontsize=8)
    axes[1].axis('off')

    # --- Polígono aproximado ---
    axes[2].imshow(binary_img, cmap='gray', alpha=0.4)

    # Dibujar polígono
    bpx_closed = bpx + [bpx[0]]
    bpy_closed = bpy + [bpy[0]]
    axes[2].plot(bpx_closed, bpy_closed, 'g-', linewidth=1.5, label='Polígono')
    axes[2].plot(bpx, bpy, 'ro', markersize=5)

    # Numerar los dominant points
    for idx_dp, (px, py) in enumerate(zip(bpx, bpy)):
        axes[2].annotate(str(idx_dp+1), (px, py),
                        textcoords="offset points", xytext=(3, 3),
                        fontsize=6, color='yellow')

    axes[2].set_title(f'Polígono aproximado\n(DP={len(break_points_indices)})')
    axes[2].legend(fontsize=8)
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    # Métricas finales
    def ise_total(contorno, bp_list):
        total = 0
        m = len(bp_list)
        for i in range(m):
            total += calcular_ise_segmento(
                contorno, bp_list[i], bp_list[(i+1) % m])
        return total

    n = len(contorno)
    dp = len(break_points_indices)
    ise = ise_total(contorno, break_points_indices)
    cr  = n / dp if dp > 0 else 0
    fom = n / (dp * ise) if (dp * ise) > 0 else float('inf')

    print("\n" + "="*45)
    print("   CRITERIOS DE ERROR")
    print("="*45)
    print(f"  n  (tamaño contorno):  {n}")
    print(f"  DP (puntos dominantes):{dp}")
    print(f"  ISE:                   {ise:.4f}")
    print(f"  CR  = n/DP:            {cr:.3f}")
    print(f"  FOM = n/(DP·ISE):      {fom:.4f}")
    print("="*45)


# =============================================
# 9. MAIN
# =============================================

if __name__ == "__main__":

    ruta = r"C:\Users\delga\Documents\IMAGE_PRUEBA\bottle-07.gif"

    # Tolerancia T — ajusta este valor:
    # T pequeño = más break points, más detalle
    # T grande  = menos break points, más simplificado
    T = 50.0

    print(f"\nCargando imagen: {ruta}")
    binary_img = cargar_imagen_binaria(ruta)
    print(f"Tamaño: {binary_img.shape}")

    print("\nExtrayendo contorno...")
    contorno = extraer_contorno_ordenado(binary_img)
    print(f"Puntos en contorno: {len(contorno)}")

    if len(contorno) < 4:
        print("ERROR: contorno muy pequeño, revisa la imagen.")
    else:
        print(f"\nEjecutando Algorithm 1 (T={T})...")
        bps_finales = aproximacion_poligonal(contorno, T)

        print(f"\nDominant Points finales: {len(bps_finales)}")
        visualizar_resultado(binary_img, contorno, bps_finales, T, ruta)
