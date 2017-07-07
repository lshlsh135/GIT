# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 07:48:34 2017

https://tensorflowkorea.gitbooks.io/tensorflow-kr/content/g3doc/get_started/basic_usage.html
 
@author: SH-NoteBook
"""

import tensorflow as tf

# 1x2 행렬을 만드는 constant op을 만들어 봅시다.
# 이 op는 default graph에 노드로 들어갈 것입니다.
# Create a constant op that produces a 1x2 matrix.  The op is
# added as a node to the default graph.
#
# 생성함수에서 나온 값은 constant op의 결과값입니다.
# The value returned by the constructor represents the output
# of the constant op.
matrix1 = tf.constant([[3., 3.]])

# 2x1 행렬을 만드는 constant op을 만들어봅시다.
# Create another Constant that produces a 2x1 matrix.
matrix2 = tf.constant([[2.],[2.]])

# 'matrix1'과 'matrix2를 입력값으로 하는 Matmul op(역자 주: 행렬곱 op)을
# 만들어 봅시다.
# 이 op의 결과값인 'product'는 행렬곱의 결과를 의미합니다.
# Create a Matmul op that takes 'matrix1' and 'matrix2' as inputs.
# The returned value, 'product', represents the result of the matrix
# multiplication.
product = tf.matmul(matrix1, matrix2)


# default graph를 실행시켜 봅시다.
# Launch the default graph.
sess = tf.Session()

# 행렬곱 작업(op)을 실행하기 위해 session의 'run()' 메서드를 호출해서 행렬곱 
# 작업의 결과값인 'product' 값을 넘겨줍시다. 그 결과값을 원한다는 뜻입니다.
# To run the matmul op we call the session 'run()' method, passing 'product'
# which represents the output of the matmul op.  This indicates to the call
# that we want to get the output of the matmul op back.
#
# 작업에 필요한 모든 입력값들은 자동적으로 session에서 실행되며 보통은 병렬로 
# 처리됩니다.
# All inputs needed by the op are run automatically by the session.  They
# typically are run in parallel.
#
# 'run(product)'가 호출되면 op 3개가 실행됩니다. 2개는 상수고 1개는 행렬곱이죠.
# The call 'run(product)' thus causes the execution of three ops in the
# graph: the two constants and matmul.
#
# 작업의 결과물은 numpy `ndarray` 오브젝트인 result' 값으로 나옵니다.
# The output of the op is returned in 'result' as a numpy `ndarray` object.
result = sess.run(product)
print(result)
# ==> [[ 12.]]

# 실행을 마치면 Session을 닫읍시다.
# Close the Session when we're done.
sess.close()

#연산에 쓰인 시스템 자원을 돌려보내려면 session을 닫아야 합니다. 시스템 자원을 더 쉽게 관리하려면 with 구문을 쓰면 됩니다. 각 Session에 컨텍스트 매니저(역자 주: 파이썬의 요소 중 하나로 주로 'with' 구문에서 쓰임)가 있어서 'with' 구문 블락의 끝에서 자동으로 'close()'가 호출됩니다.
with tf.Session() as sess:
  result = sess.run([product])
  print(result)
  
# 인터렉티브 TensorFlow Session을 시작해봅시다.
# Enter an interactive TensorFlow Session.
import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0])

# 초기화 op의 run() 메서드를 이용해서 'x'를 초기화합시다.
# Initialize 'x' using the run() method of its initializer op.
x.initializer.run()

# 'x'에서 'a'를 빼는 작업을 추가하고 실행시켜서 결과를 봅시다.
# Add an op to subtract 'a' from 'x'.  Run it and print the result
sub = tf.subtract(x, a)
print(sub.eval())
# ==> [-2. -1.]

# 실행을 마치면 Session을 닫읍시다.
# Close the Session when we're done.
sess.close()

# 값이 0인 스칼라로 초기화된 변수를 만듭니다.
# Create a Variable, that will be initialized to the scalar value 0.
state = tf.Variable(0, name="counter")

# 'state'에 1을 더하는 작업(op)을 만듭니다.
# Create an Op to add one to `state`.

one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# 그래프를 한 번 작동시킨 후에는 'init' 작업(op)을 실행해서 변수를 초기화해야
# 합니다. 먼저 'init' 작업(op)을 추가해 봅시다.
# Variables must be initialized by running an `init` Op after having
# launched the graph.  We first have to add the `init` Op to the graph.
init_op = tf.global_variables_initializer()

# graph와 작업(op)들을 실행시킵니다.
# Launch the graph and run the ops.
with tf.Session() as sess:
  # 'init' 작업(op)을 실행합니다.
  # Run the 'init' op
  sess.run(init_op)
  # 'state'의 시작값을 출력합니다.
  # Print the initial value of 'state'
  print(sess.run(state))
  # 'state'값을 업데이트하고 출력하는 작업(op)을 실행합니다.
  # Run the op that updates 'state' and print 'state'.
  for _ in range(3):
    sess.run(update)
    print(sess.run(state))

# output:

# 0
# 1
# 2
# 3

#==============================================================================
# Fetches
#==============================================================================
input1 = tf.constant([3.0])
input2 = tf.constant([2.0])
input3 = tf.constant([5.0])
intermed = tf.add(input2, input3)
mul = tf.multiply(input1, intermed)

with tf.Session() as sess:
  result = sess.run([mul, intermed])
  print(result)

# output:
# [array([ 21.], dtype=float32), array([ 7.], dtype=float32)]

#==============================================================================
#Feeds 
#==============================================================================






