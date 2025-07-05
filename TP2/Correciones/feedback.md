El TP tiene unos cuantos problemas que tienen que corregir para reentregar. También aparecieron problemas nuevos en el TP1.
Acá abajo les dejo un punteo. En el notebook y en el código también les dejé comentarios que empiezan con "VALEN:" y "#>".

Si tienen consultas para la reentrega, la semana que viene voy a estar por el discord para responder cosas. Voy a tratar
de estar disponible el martes y seguro el miércoles. Los otros días no aseguro nada.
Si quieren me pueden arrobar por discord para ver si estoy y nos metemos en una sala para ver lo que necesiten.
Por si no están en el discord de la materia, este es el link (vence en una semana): https://discord.gg/6RDuxRPX

Cosas importantes:

- En el 1a está rota la demo para R y mezclan notaciones que no tienen mucho sentido. Hay que arreglar eso.

- En el 2b tienen que demostrar que los autovalores de la inversa son los inversos y que están asociados al mismo autovector.

- Falta el 2c.

Detalles:

- En el TP1 agregaron las funciones para invertir matrices diagonales y con LU pero no las reemplazaron en el código. Borren
  las viejas y pongan esas (y obviamente verifiquen que no se rompe nada con el cambio).

# Pendientes Gallar

## Pendiente, arreglado, queda chequear martes

- Falta la parte del notebook del 3. También tienen la implementación de modularidad_iterativo mal, les dejé un comentario en el código.

## Same anterior

- Toda la parte de modularidad del 4 está mal porque la implementación está mal, vuelvan a eso después de arreglarlo. Después,
  la idea no es que prueben varios niveles del laplaciano por cada m, sino que traten de aproximarse lo mejor posible a la cantidad que
  hace modularidad. También les dejé un comentario en el notebook sobre eso. Estos arreglos seguramente afecten a las conclusiones del 5.

# Solucionados

## Solucionado, mismo resultado

- En el TP1 rompieron calcula_matriz_C_continua y por eso ahora el 5 y el 6 les da mal. Estaría bueno que lo arreglen manteniendo
  las cosas con numpy, pero si es mucho quilombo pueden volver a dejar el código que tenían en la primer entrega que
  andaba bien. Fíjense que los resultados vuelvan a dar lo mismo que en la primer entrega.

## Solucionado, ya se invirtio el signo

- Tienen un bug en el código de metpotI2 (se los marqué en el código) y por eso el autovalor les está dando mal y el mu
  está afectando al resultado.

## Solucionado, nada mas faltaba hacer la funcion de grafico

- En el 3 del TP1 borraron los mapas del alfa. Los tenían hechos en la primer entrega así que no me queda del todo claro
  por que los mataron. Vuelvan a ponerlos (y acuérdense de juntarlos como hicieron con los de m).
