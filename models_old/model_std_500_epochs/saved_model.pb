¿
¬ý
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
*
Erf
x"T
y"T"
Ttype:
2
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
-
Sqrt
x"T
y"T"
Ttype:

2
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68Úí
`
meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namemean
Y
mean/Read/ReadVariableOpReadVariableOpmean*
_output_shapes
:*
dtype0
h
varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
variance
a
variance/Read/ReadVariableOpReadVariableOpvariance*
_output_shapes
:*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0	
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:*
dtype0

predictions/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*#
shared_namepredictions/kernel
z
&predictions/kernel/Read/ReadVariableOpReadVariableOppredictions/kernel*
_output_shapes
:	*
dtype0
x
predictions/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namepredictions/bias
q
$predictions/bias/Read/ReadVariableOpReadVariableOppredictions/bias*
_output_shapes
:*
dtype0
f
ConstConst*
_output_shapes

:*
dtype0*)
value B"f¢¼ge»XD;â·?
h
Const_1Const*
_output_shapes

:*
dtype0*)
value B"Í^<?·Ô9?ý¥?Ê>@

NoOpNoOp
«
Const_2Const"/device:CPU:0*
_output_shapes
: *
dtype0*ä
valueÚB× BÐ
¿
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature

signatures*

_init_input_shape* 
¾

_keep_axis
_reduce_axis
_reduce_axis_mask
_broadcast_shape
mean

adapt_mean
variance
adapt_variance
	count
	keras_api
_adapt_function*
¦

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
¦

kernel
 bias
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses*
5
0
1
2
3
4
5
 6*
 
0
1
2
 3*
* 
°
'non_trainable_variables

(layers
)metrics
*layer_regularization_losses
+layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*
* 
* 
* 

,serving_default* 
* 
* 
* 
* 
* 
RL
VARIABLE_VALUEmean4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEvariance8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEcount5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

-non_trainable_variables

.layers
/metrics
0layer_regularization_losses
1layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
b\
VARIABLE_VALUEpredictions/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEpredictions/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
 1*

0
 1*
* 

2non_trainable_variables

3layers
4metrics
5layer_regularization_losses
6layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*
* 
* 

0
1
2*
 
0
1
2
3*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
z
serving_default_input_2Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2ConstConst_1dense_1/kerneldense_1/biaspredictions/kernelpredictions/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *0
f+R)
'__inference_signature_wrapper_442077728
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
z
StaticRegexFullMatchStaticRegexFullMatchsaver_filename"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*
\
Const_3Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part
a
Const_4Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
h
SelectSelectStaticRegexFullMatchConst_3Const_4"/device:CPU:**
T0*
_output_shapes
: 
`

StringJoin
StringJoinsaver_filenameSelect"/device:CPU:**
N*
_output_shapes
: 
L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
x
ShardedFilenameShardedFilename
StringJoinShardedFilename/shard
num_shards"/device:CPU:0*
_output_shapes
: 

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*¶
value¬B©B4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
}
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*#
valueBB B B B B B B B 
æ
SaveV2SaveV2ShardedFilenameSaveV2/tensor_namesSaveV2/shape_and_slicesmean/Read/ReadVariableOpvariance/Read/ReadVariableOpcount/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp&predictions/kernel/Read/ReadVariableOp$predictions/bias/Read/ReadVariableOpConst_2"/device:CPU:0*
dtypes

2	

&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
o
MergeV2CheckpointsMergeV2Checkpoints&MergeV2Checkpoints/checkpoint_prefixessaver_filename"/device:CPU:0
i
IdentityIdentitysaver_filename^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*¶
value¬B©B4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH

RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*#
valueBB B B B B B B B 
·
	RestoreV2	RestoreV2saver_filenameRestoreV2/tensor_namesRestoreV2/shape_and_slices"/device:CPU:0*4
_output_shapes"
 ::::::::*
dtypes

2	
S

Identity_1Identity	RestoreV2"/device:CPU:0*
T0*
_output_shapes
:
R
AssignVariableOpAssignVariableOpmean
Identity_1"/device:CPU:0*
dtype0
U

Identity_2IdentityRestoreV2:1"/device:CPU:0*
T0*
_output_shapes
:
X
AssignVariableOp_1AssignVariableOpvariance
Identity_2"/device:CPU:0*
dtype0
U

Identity_3IdentityRestoreV2:2"/device:CPU:0*
T0	*
_output_shapes
:
U
AssignVariableOp_2AssignVariableOpcount
Identity_3"/device:CPU:0*
dtype0	
U

Identity_4IdentityRestoreV2:3"/device:CPU:0*
T0*
_output_shapes
:
^
AssignVariableOp_3AssignVariableOpdense_1/kernel
Identity_4"/device:CPU:0*
dtype0
U

Identity_5IdentityRestoreV2:4"/device:CPU:0*
T0*
_output_shapes
:
\
AssignVariableOp_4AssignVariableOpdense_1/bias
Identity_5"/device:CPU:0*
dtype0
U

Identity_6IdentityRestoreV2:5"/device:CPU:0*
T0*
_output_shapes
:
b
AssignVariableOp_5AssignVariableOppredictions/kernel
Identity_6"/device:CPU:0*
dtype0
U

Identity_7IdentityRestoreV2:6"/device:CPU:0*
T0*
_output_shapes
:
`
AssignVariableOp_6AssignVariableOppredictions/bias
Identity_7"/device:CPU:0*
dtype0

NoOp_1NoOp"/device:CPU:0
ð

Identity_8Identitysaver_filename^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^NoOp_1"/device:CPU:0*
T0*
_output_shapes
: ¡
+
¬
$__inference__wrapped_model_442077163
input_2
model_1_norm_sub_y
model_1_norm_sqrt_xA
.model_1_dense_1_matmul_readvariableop_resource:	>
/model_1_dense_1_biasadd_readvariableop_resource:	E
2model_1_predictions_matmul_readvariableop_resource:	A
3model_1_predictions_biasadd_readvariableop_resource:
identity¢&model_1/dense_1/BiasAdd/ReadVariableOp¢%model_1/dense_1/MatMul/ReadVariableOp¢*model_1/predictions/BiasAdd/ReadVariableOp¢)model_1/predictions/MatMul/ReadVariableOpf
model_1/norm/subSubinput_2model_1_norm_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
model_1/norm/SqrtSqrtmodel_1_norm_sqrt_x*
T0*
_output_shapes

:[
model_1/norm/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
model_1/norm/MaximumMaximummodel_1/norm/Sqrt:y:0model_1/norm/Maximum/y:output:0*
T0*
_output_shapes

:
model_1/norm/truedivRealDivmodel_1/norm/sub:z:0model_1/norm/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%model_1/dense_1/MatMul/ReadVariableOpReadVariableOp.model_1_dense_1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
model_1/dense_1/MatMulMatMulmodel_1/norm/truediv:z:0-model_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&model_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0§
model_1/dense_1/BiasAddBiasAdd model_1/dense_1/MatMul:product:0.model_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
model_1/dense_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
model_1/dense_1/Gelu/mulMul#model_1/dense_1/Gelu/mul/x:output:0 model_1/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
model_1/dense_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?¢
model_1/dense_1/Gelu/truedivRealDiv model_1/dense_1/BiasAdd:output:0$model_1/dense_1/Gelu/Cast/x:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
model_1/dense_1/Gelu/ErfErf model_1/dense_1/Gelu/truediv:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
model_1/dense_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model_1/dense_1/Gelu/addAddV2#model_1/dense_1/Gelu/add/x:output:0model_1/dense_1/Gelu/Erf:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model_1/dense_1/Gelu/mul_1Mulmodel_1/dense_1/Gelu/mul:z:0model_1/dense_1/Gelu/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)model_1/predictions/MatMul/ReadVariableOpReadVariableOp2model_1_predictions_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0©
model_1/predictions/MatMulMatMulmodel_1/dense_1/Gelu/mul_1:z:01model_1/predictions/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*model_1/predictions/BiasAdd/ReadVariableOpReadVariableOp3model_1_predictions_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0²
model_1/predictions/BiasAddBiasAdd$model_1/predictions/MatMul:product:02model_1/predictions/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
model_1/predictions/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¤
model_1/predictions/Gelu/mulMul'model_1/predictions/Gelu/mul/x:output:0$model_1/predictions/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
model_1/predictions/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?­
 model_1/predictions/Gelu/truedivRealDiv$model_1/predictions/BiasAdd:output:0(model_1/predictions/Gelu/Cast/x:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
model_1/predictions/Gelu/ErfErf$model_1/predictions/Gelu/truediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
model_1/predictions/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¢
model_1/predictions/Gelu/addAddV2'model_1/predictions/Gelu/add/x:output:0 model_1/predictions/Gelu/Erf:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model_1/predictions/Gelu/mul_1Mul model_1/predictions/Gelu/mul:z:0 model_1/predictions/Gelu/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
IdentityIdentity"model_1/predictions/Gelu/mul_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
NoOpNoOp'^model_1/dense_1/BiasAdd/ReadVariableOp&^model_1/dense_1/MatMul/ReadVariableOp+^model_1/predictions/BiasAdd/ReadVariableOp*^model_1/predictions/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ::: : : : 2P
&model_1/dense_1/BiasAdd/ReadVariableOp&model_1/dense_1/BiasAdd/ReadVariableOp2N
%model_1/dense_1/MatMul/ReadVariableOp%model_1/dense_1/MatMul/ReadVariableOp2X
*model_1/predictions/BiasAdd/ReadVariableOp*model_1/predictions/BiasAdd/ReadVariableOp2V
)model_1/predictions/MatMul/ReadVariableOp)model_1/predictions/MatMul/ReadVariableOp:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2:$ 

_output_shapes

::$ 

_output_shapes

:
û%
þ
F__inference_model_1_layer_call_and_return_conditional_losses_442077514
input_2

norm_sub_y
norm_sqrt_x9
&dense_1_matmul_readvariableop_resource:	6
'dense_1_biasadd_readvariableop_resource:	=
*predictions_matmul_readvariableop_resource:	9
+predictions_biasadd_readvariableop_resource:
identity¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢"predictions/BiasAdd/ReadVariableOp¢!predictions/MatMul/ReadVariableOpV
norm/subSubinput_2
norm_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
	norm/SqrtSqrtnorm_sqrt_x*
T0*
_output_shapes

:S
norm/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3h
norm/MaximumMaximumnorm/Sqrt:y:0norm/Maximum/y:output:0*
T0*
_output_shapes

:i
norm/truedivRealDivnorm/sub:z:0norm/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_1/MatMulMatMulnorm/truediv:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
dense_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
dense_1/Gelu/mulMuldense_1/Gelu/mul/x:output:0dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
dense_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?
dense_1/Gelu/truedivRealDivdense_1/BiasAdd:output:0dense_1/Gelu/Cast/x:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_1/Gelu/ErfErfdense_1/Gelu/truediv:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
dense_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
dense_1/Gelu/addAddV2dense_1/Gelu/add/x:output:0dense_1/Gelu/Erf:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
dense_1/Gelu/mul_1Muldense_1/Gelu/mul:z:0dense_1/Gelu/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!predictions/MatMul/ReadVariableOpReadVariableOp*predictions_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
predictions/MatMulMatMuldense_1/Gelu/mul_1:z:0)predictions/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"predictions/BiasAdd/ReadVariableOpReadVariableOp+predictions_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
predictions/BiasAddBiasAddpredictions/MatMul:product:0*predictions/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
predictions/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
predictions/Gelu/mulMulpredictions/Gelu/mul/x:output:0predictions/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
predictions/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?
predictions/Gelu/truedivRealDivpredictions/BiasAdd:output:0 predictions/Gelu/Cast/x:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
predictions/Gelu/ErfErfpredictions/Gelu/truediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
predictions/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
predictions/Gelu/addAddV2predictions/Gelu/add/x:output:0predictions/Gelu/Erf:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
predictions/Gelu/mul_1Mulpredictions/Gelu/mul:z:0predictions/Gelu/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitypredictions/Gelu/mul_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp#^predictions/BiasAdd/ReadVariableOp"^predictions/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ::: : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2H
"predictions/BiasAdd/ReadVariableOp"predictions/BiasAdd/ReadVariableOp2F
!predictions/MatMul/ReadVariableOp!predictions/MatMul/ReadVariableOp:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2:$ 

_output_shapes

::$ 

_output_shapes

:
Ý%
â
+__inference_model_1_layer_call_fn_442077631

inputs

norm_sub_y
norm_sqrt_x9
&dense_1_matmul_readvariableop_resource:	6
'dense_1_biasadd_readvariableop_resource:	=
*predictions_matmul_readvariableop_resource:	9
+predictions_biasadd_readvariableop_resource:
identity¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢"predictions/BiasAdd/ReadVariableOp¢!predictions/MatMul/ReadVariableOpU
norm/subSubinputs
norm_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
	norm/SqrtSqrtnorm_sqrt_x*
T0*
_output_shapes

:S
norm/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3h
norm/MaximumMaximumnorm/Sqrt:y:0norm/Maximum/y:output:0*
T0*
_output_shapes

:i
norm/truedivRealDivnorm/sub:z:0norm/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_1/MatMulMatMulnorm/truediv:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
dense_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
dense_1/Gelu/mulMuldense_1/Gelu/mul/x:output:0dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
dense_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?
dense_1/Gelu/truedivRealDivdense_1/BiasAdd:output:0dense_1/Gelu/Cast/x:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_1/Gelu/ErfErfdense_1/Gelu/truediv:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
dense_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
dense_1/Gelu/addAddV2dense_1/Gelu/add/x:output:0dense_1/Gelu/Erf:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
dense_1/Gelu/mul_1Muldense_1/Gelu/mul:z:0dense_1/Gelu/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!predictions/MatMul/ReadVariableOpReadVariableOp*predictions_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
predictions/MatMulMatMuldense_1/Gelu/mul_1:z:0)predictions/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"predictions/BiasAdd/ReadVariableOpReadVariableOp+predictions_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
predictions/BiasAddBiasAddpredictions/MatMul:product:0*predictions/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
predictions/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
predictions/Gelu/mulMulpredictions/Gelu/mul/x:output:0predictions/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
predictions/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?
predictions/Gelu/truedivRealDivpredictions/BiasAdd:output:0 predictions/Gelu/Cast/x:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
predictions/Gelu/ErfErfpredictions/Gelu/truediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
predictions/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
predictions/Gelu/addAddV2predictions/Gelu/add/x:output:0predictions/Gelu/Erf:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
predictions/Gelu/mul_1Mulpredictions/Gelu/mul:z:0predictions/Gelu/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitypredictions/Gelu/mul_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp#^predictions/BiasAdd/ReadVariableOp"^predictions/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ::: : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2H
"predictions/BiasAdd/ReadVariableOp"predictions/BiasAdd/ReadVariableOp2F
!predictions/MatMul/ReadVariableOp!predictions/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
Ý%
â
+__inference_model_1_layer_call_fn_442077592

inputs

norm_sub_y
norm_sqrt_x9
&dense_1_matmul_readvariableop_resource:	6
'dense_1_biasadd_readvariableop_resource:	=
*predictions_matmul_readvariableop_resource:	9
+predictions_biasadd_readvariableop_resource:
identity¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢"predictions/BiasAdd/ReadVariableOp¢!predictions/MatMul/ReadVariableOpU
norm/subSubinputs
norm_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
	norm/SqrtSqrtnorm_sqrt_x*
T0*
_output_shapes

:S
norm/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3h
norm/MaximumMaximumnorm/Sqrt:y:0norm/Maximum/y:output:0*
T0*
_output_shapes

:i
norm/truedivRealDivnorm/sub:z:0norm/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_1/MatMulMatMulnorm/truediv:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
dense_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
dense_1/Gelu/mulMuldense_1/Gelu/mul/x:output:0dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
dense_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?
dense_1/Gelu/truedivRealDivdense_1/BiasAdd:output:0dense_1/Gelu/Cast/x:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_1/Gelu/ErfErfdense_1/Gelu/truediv:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
dense_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
dense_1/Gelu/addAddV2dense_1/Gelu/add/x:output:0dense_1/Gelu/Erf:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
dense_1/Gelu/mul_1Muldense_1/Gelu/mul:z:0dense_1/Gelu/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!predictions/MatMul/ReadVariableOpReadVariableOp*predictions_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
predictions/MatMulMatMuldense_1/Gelu/mul_1:z:0)predictions/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"predictions/BiasAdd/ReadVariableOpReadVariableOp+predictions_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
predictions/BiasAddBiasAddpredictions/MatMul:product:0*predictions/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
predictions/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
predictions/Gelu/mulMulpredictions/Gelu/mul/x:output:0predictions/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
predictions/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?
predictions/Gelu/truedivRealDivpredictions/BiasAdd:output:0 predictions/Gelu/Cast/x:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
predictions/Gelu/ErfErfpredictions/Gelu/truediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
predictions/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
predictions/Gelu/addAddV2predictions/Gelu/add/x:output:0predictions/Gelu/Erf:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
predictions/Gelu/mul_1Mulpredictions/Gelu/mul:z:0predictions/Gelu/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitypredictions/Gelu/mul_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp#^predictions/BiasAdd/ReadVariableOp"^predictions/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ::: : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2H
"predictions/BiasAdd/ReadVariableOp"predictions/BiasAdd/ReadVariableOp2F
!predictions/MatMul/ReadVariableOp!predictions/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
à%
ã
+__inference_model_1_layer_call_fn_442077203
input_2

norm_sub_y
norm_sqrt_x9
&dense_1_matmul_readvariableop_resource:	6
'dense_1_biasadd_readvariableop_resource:	=
*predictions_matmul_readvariableop_resource:	9
+predictions_biasadd_readvariableop_resource:
identity¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢"predictions/BiasAdd/ReadVariableOp¢!predictions/MatMul/ReadVariableOpV
norm/subSubinput_2
norm_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
	norm/SqrtSqrtnorm_sqrt_x*
T0*
_output_shapes

:S
norm/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3h
norm/MaximumMaximumnorm/Sqrt:y:0norm/Maximum/y:output:0*
T0*
_output_shapes

:i
norm/truedivRealDivnorm/sub:z:0norm/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_1/MatMulMatMulnorm/truediv:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
dense_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
dense_1/Gelu/mulMuldense_1/Gelu/mul/x:output:0dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
dense_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?
dense_1/Gelu/truedivRealDivdense_1/BiasAdd:output:0dense_1/Gelu/Cast/x:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_1/Gelu/ErfErfdense_1/Gelu/truediv:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
dense_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
dense_1/Gelu/addAddV2dense_1/Gelu/add/x:output:0dense_1/Gelu/Erf:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
dense_1/Gelu/mul_1Muldense_1/Gelu/mul:z:0dense_1/Gelu/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!predictions/MatMul/ReadVariableOpReadVariableOp*predictions_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
predictions/MatMulMatMuldense_1/Gelu/mul_1:z:0)predictions/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"predictions/BiasAdd/ReadVariableOpReadVariableOp+predictions_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
predictions/BiasAddBiasAddpredictions/MatMul:product:0*predictions/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
predictions/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
predictions/Gelu/mulMulpredictions/Gelu/mul/x:output:0predictions/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
predictions/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?
predictions/Gelu/truedivRealDivpredictions/BiasAdd:output:0 predictions/Gelu/Cast/x:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
predictions/Gelu/ErfErfpredictions/Gelu/truediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
predictions/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
predictions/Gelu/addAddV2predictions/Gelu/add/x:output:0predictions/Gelu/Erf:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
predictions/Gelu/mul_1Mulpredictions/Gelu/mul:z:0predictions/Gelu/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitypredictions/Gelu/mul_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp#^predictions/BiasAdd/ReadVariableOp"^predictions/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ::: : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2H
"predictions/BiasAdd/ReadVariableOp"predictions/BiasAdd/ReadVariableOp2F
!predictions/MatMul/ReadVariableOp!predictions/MatMul/ReadVariableOp:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2:$ 

_output_shapes

::$ 

_output_shapes

:
à%
ã
+__inference_model_1_layer_call_fn_442077475
input_2

norm_sub_y
norm_sqrt_x9
&dense_1_matmul_readvariableop_resource:	6
'dense_1_biasadd_readvariableop_resource:	=
*predictions_matmul_readvariableop_resource:	9
+predictions_biasadd_readvariableop_resource:
identity¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢"predictions/BiasAdd/ReadVariableOp¢!predictions/MatMul/ReadVariableOpV
norm/subSubinput_2
norm_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
	norm/SqrtSqrtnorm_sqrt_x*
T0*
_output_shapes

:S
norm/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3h
norm/MaximumMaximumnorm/Sqrt:y:0norm/Maximum/y:output:0*
T0*
_output_shapes

:i
norm/truedivRealDivnorm/sub:z:0norm/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_1/MatMulMatMulnorm/truediv:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
dense_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
dense_1/Gelu/mulMuldense_1/Gelu/mul/x:output:0dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
dense_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?
dense_1/Gelu/truedivRealDivdense_1/BiasAdd:output:0dense_1/Gelu/Cast/x:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_1/Gelu/ErfErfdense_1/Gelu/truediv:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
dense_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
dense_1/Gelu/addAddV2dense_1/Gelu/add/x:output:0dense_1/Gelu/Erf:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
dense_1/Gelu/mul_1Muldense_1/Gelu/mul:z:0dense_1/Gelu/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!predictions/MatMul/ReadVariableOpReadVariableOp*predictions_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
predictions/MatMulMatMuldense_1/Gelu/mul_1:z:0)predictions/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"predictions/BiasAdd/ReadVariableOpReadVariableOp+predictions_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
predictions/BiasAddBiasAddpredictions/MatMul:product:0*predictions/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
predictions/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
predictions/Gelu/mulMulpredictions/Gelu/mul/x:output:0predictions/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
predictions/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?
predictions/Gelu/truedivRealDivpredictions/BiasAdd:output:0 predictions/Gelu/Cast/x:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
predictions/Gelu/ErfErfpredictions/Gelu/truediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
predictions/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
predictions/Gelu/addAddV2predictions/Gelu/add/x:output:0predictions/Gelu/Erf:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
predictions/Gelu/mul_1Mulpredictions/Gelu/mul:z:0predictions/Gelu/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitypredictions/Gelu/mul_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp#^predictions/BiasAdd/ReadVariableOp"^predictions/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ::: : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2H
"predictions/BiasAdd/ReadVariableOp"predictions/BiasAdd/ReadVariableOp2F
!predictions/MatMul/ReadVariableOp!predictions/MatMul/ReadVariableOp:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2:$ 

_output_shapes

::$ 

_output_shapes

:
Â
ù
F__inference_dense_1_layer_call_and_return_conditional_losses_442077764

inputs1
matmul_readvariableop_resource:	.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿO

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?i
Gelu/mulMulGelu/mul/x:output:0BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?r
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
Gelu/ErfErfGelu/truediv:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿO

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?g
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
IdentityIdentityGelu/mul_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ø%
ý
F__inference_model_1_layer_call_and_return_conditional_losses_442077670

inputs

norm_sub_y
norm_sqrt_x9
&dense_1_matmul_readvariableop_resource:	6
'dense_1_biasadd_readvariableop_resource:	=
*predictions_matmul_readvariableop_resource:	9
+predictions_biasadd_readvariableop_resource:
identity¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢"predictions/BiasAdd/ReadVariableOp¢!predictions/MatMul/ReadVariableOpU
norm/subSubinputs
norm_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
	norm/SqrtSqrtnorm_sqrt_x*
T0*
_output_shapes

:S
norm/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3h
norm/MaximumMaximumnorm/Sqrt:y:0norm/Maximum/y:output:0*
T0*
_output_shapes

:i
norm/truedivRealDivnorm/sub:z:0norm/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_1/MatMulMatMulnorm/truediv:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
dense_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
dense_1/Gelu/mulMuldense_1/Gelu/mul/x:output:0dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
dense_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?
dense_1/Gelu/truedivRealDivdense_1/BiasAdd:output:0dense_1/Gelu/Cast/x:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_1/Gelu/ErfErfdense_1/Gelu/truediv:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
dense_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
dense_1/Gelu/addAddV2dense_1/Gelu/add/x:output:0dense_1/Gelu/Erf:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
dense_1/Gelu/mul_1Muldense_1/Gelu/mul:z:0dense_1/Gelu/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!predictions/MatMul/ReadVariableOpReadVariableOp*predictions_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
predictions/MatMulMatMuldense_1/Gelu/mul_1:z:0)predictions/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"predictions/BiasAdd/ReadVariableOpReadVariableOp+predictions_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
predictions/BiasAddBiasAddpredictions/MatMul:product:0*predictions/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
predictions/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
predictions/Gelu/mulMulpredictions/Gelu/mul/x:output:0predictions/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
predictions/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?
predictions/Gelu/truedivRealDivpredictions/BiasAdd:output:0 predictions/Gelu/Cast/x:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
predictions/Gelu/ErfErfpredictions/Gelu/truediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
predictions/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
predictions/Gelu/addAddV2predictions/Gelu/add/x:output:0predictions/Gelu/Erf:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
predictions/Gelu/mul_1Mulpredictions/Gelu/mul:z:0predictions/Gelu/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitypredictions/Gelu/mul_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp#^predictions/BiasAdd/ReadVariableOp"^predictions/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ::: : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2H
"predictions/BiasAdd/ReadVariableOp"predictions/BiasAdd/ReadVariableOp2F
!predictions/MatMul/ReadVariableOp!predictions/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
§
Þ
+__inference_dense_1_layer_call_fn_442077746

inputs1
matmul_readvariableop_resource:	.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿO

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?i
Gelu/mulMulGelu/mul/x:output:0BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?r
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
Gelu/ErfErfGelu/truediv:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿO

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?g
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
IdentityIdentityGelu/mul_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
û%
þ
F__inference_model_1_layer_call_and_return_conditional_losses_442077553
input_2

norm_sub_y
norm_sqrt_x9
&dense_1_matmul_readvariableop_resource:	6
'dense_1_biasadd_readvariableop_resource:	=
*predictions_matmul_readvariableop_resource:	9
+predictions_biasadd_readvariableop_resource:
identity¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢"predictions/BiasAdd/ReadVariableOp¢!predictions/MatMul/ReadVariableOpV
norm/subSubinput_2
norm_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
	norm/SqrtSqrtnorm_sqrt_x*
T0*
_output_shapes

:S
norm/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3h
norm/MaximumMaximumnorm/Sqrt:y:0norm/Maximum/y:output:0*
T0*
_output_shapes

:i
norm/truedivRealDivnorm/sub:z:0norm/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_1/MatMulMatMulnorm/truediv:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
dense_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
dense_1/Gelu/mulMuldense_1/Gelu/mul/x:output:0dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
dense_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?
dense_1/Gelu/truedivRealDivdense_1/BiasAdd:output:0dense_1/Gelu/Cast/x:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_1/Gelu/ErfErfdense_1/Gelu/truediv:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
dense_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
dense_1/Gelu/addAddV2dense_1/Gelu/add/x:output:0dense_1/Gelu/Erf:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
dense_1/Gelu/mul_1Muldense_1/Gelu/mul:z:0dense_1/Gelu/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!predictions/MatMul/ReadVariableOpReadVariableOp*predictions_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
predictions/MatMulMatMuldense_1/Gelu/mul_1:z:0)predictions/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"predictions/BiasAdd/ReadVariableOpReadVariableOp+predictions_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
predictions/BiasAddBiasAddpredictions/MatMul:product:0*predictions/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
predictions/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
predictions/Gelu/mulMulpredictions/Gelu/mul/x:output:0predictions/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
predictions/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?
predictions/Gelu/truedivRealDivpredictions/BiasAdd:output:0 predictions/Gelu/Cast/x:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
predictions/Gelu/ErfErfpredictions/Gelu/truediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
predictions/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
predictions/Gelu/addAddV2predictions/Gelu/add/x:output:0predictions/Gelu/Erf:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
predictions/Gelu/mul_1Mulpredictions/Gelu/mul:z:0predictions/Gelu/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitypredictions/Gelu/mul_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp#^predictions/BiasAdd/ReadVariableOp"^predictions/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ::: : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2H
"predictions/BiasAdd/ReadVariableOp"predictions/BiasAdd/ReadVariableOp2F
!predictions/MatMul/ReadVariableOp!predictions/MatMul/ReadVariableOp:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2:$ 

_output_shapes

::$ 

_output_shapes

:
¾
ü
J__inference_predictions_layer_call_and_return_conditional_losses_442077800

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿO

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?h
Gelu/mulMulGelu/mul/x:output:0BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?q
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
Gelu/ErfErfGelu/truediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿO

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?f
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
IdentityIdentityGelu/mul_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
£
á
/__inference_predictions_layer_call_fn_442077782

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿO

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?h
Gelu/mulMulGelu/mul/x:output:0BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?q
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
Gelu/ErfErfGelu/truediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿO

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?f
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
IdentityIdentityGelu/mul_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ø%
ý
F__inference_model_1_layer_call_and_return_conditional_losses_442077709

inputs

norm_sub_y
norm_sqrt_x9
&dense_1_matmul_readvariableop_resource:	6
'dense_1_biasadd_readvariableop_resource:	=
*predictions_matmul_readvariableop_resource:	9
+predictions_biasadd_readvariableop_resource:
identity¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢"predictions/BiasAdd/ReadVariableOp¢!predictions/MatMul/ReadVariableOpU
norm/subSubinputs
norm_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
	norm/SqrtSqrtnorm_sqrt_x*
T0*
_output_shapes

:S
norm/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3h
norm/MaximumMaximumnorm/Sqrt:y:0norm/Maximum/y:output:0*
T0*
_output_shapes

:i
norm/truedivRealDivnorm/sub:z:0norm/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_1/MatMulMatMulnorm/truediv:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
dense_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
dense_1/Gelu/mulMuldense_1/Gelu/mul/x:output:0dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
dense_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?
dense_1/Gelu/truedivRealDivdense_1/BiasAdd:output:0dense_1/Gelu/Cast/x:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_1/Gelu/ErfErfdense_1/Gelu/truediv:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
dense_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
dense_1/Gelu/addAddV2dense_1/Gelu/add/x:output:0dense_1/Gelu/Erf:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
dense_1/Gelu/mul_1Muldense_1/Gelu/mul:z:0dense_1/Gelu/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!predictions/MatMul/ReadVariableOpReadVariableOp*predictions_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
predictions/MatMulMatMuldense_1/Gelu/mul_1:z:0)predictions/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"predictions/BiasAdd/ReadVariableOpReadVariableOp+predictions_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
predictions/BiasAddBiasAddpredictions/MatMul:product:0*predictions/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
predictions/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
predictions/Gelu/mulMulpredictions/Gelu/mul/x:output:0predictions/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
predictions/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?
predictions/Gelu/truedivRealDivpredictions/BiasAdd:output:0 predictions/Gelu/Cast/x:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
predictions/Gelu/ErfErfpredictions/Gelu/truediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
predictions/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
predictions/Gelu/addAddV2predictions/Gelu/add/x:output:0predictions/Gelu/Erf:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
predictions/Gelu/mul_1Mulpredictions/Gelu/mul:z:0predictions/Gelu/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitypredictions/Gelu/mul_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp#^predictions/BiasAdd/ReadVariableOp"^predictions/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ::: : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2H
"predictions/BiasAdd/ReadVariableOp"predictions/BiasAdd/ReadVariableOp2F
!predictions/MatMul/ReadVariableOp!predictions/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
	
ì
'__inference_signature_wrapper_442077728
input_2
unknown
	unknown_0
	unknown_1:	
	unknown_2:	
	unknown_3:	
	unknown_4:
identity¢StatefulPartitionedCallì
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference__wrapped_model_442077163o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ::: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2:$ 

_output_shapes

::$ 

_output_shapes

:"Û,
saver_filename:0
Identity:0
Identity_88"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*®
serving_default
;
input_20
serving_default_input_2:0ÿÿÿÿÿÿÿÿÿ?
predictions0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:È>
Ö
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
6
_init_input_shape"
_tf_keras_input_layer
Ó

_keep_axis
_reduce_axis
_reduce_axis_mask
_broadcast_shape
mean

adapt_mean
variance
adapt_variance
	count
	keras_api
_adapt_function"
_tf_keras_layer
»

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
 bias
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses"
_tf_keras_layer
Q
0
1
2
3
4
5
 6"
trackable_list_wrapper
<
0
1
2
 3"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
'non_trainable_variables

(layers
)metrics
*layer_regularization_losses
+layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object
ú2÷
+__inference_model_1_layer_call_fn_442077203
+__inference_model_1_layer_call_fn_442077592
+__inference_model_1_layer_call_fn_442077631
+__inference_model_1_layer_call_fn_442077475À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
æ2ã
F__inference_model_1_layer_call_and_return_conditional_losses_442077670
F__inference_model_1_layer_call_and_return_conditional_losses_442077709
F__inference_model_1_layer_call_and_return_conditional_losses_442077514
F__inference_model_1_layer_call_and_return_conditional_losses_442077553À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÏBÌ
$__inference__wrapped_model_442077163input_2"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
,serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:2mean
:2variance
:	 2count
"
_generic_user_object
 2
²
FullArgSpec
args

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
!:	2dense_1/kernel
:2dense_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
-non_trainable_variables

.layers
/metrics
0layer_regularization_losses
1layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_dense_1_layer_call_fn_442077746¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_dense_1_layer_call_and_return_conditional_losses_442077764¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
%:#	2predictions/kernel
:2predictions/bias
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
2non_trainable_variables

3layers
4metrics
5layer_regularization_losses
6layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
Ù2Ö
/__inference_predictions_layer_call_fn_442077782¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ô2ñ
J__inference_predictions_layer_call_and_return_conditional_losses_442077800¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
5
0
1
2"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÎBË
'__inference_signature_wrapper_442077728input_2"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
	J
Const
J	
Const_1
$__inference__wrapped_model_442077163u78 0¢-
&¢#
!
input_2ÿÿÿÿÿÿÿÿÿ
ª "9ª6
4
predictions%"
predictionsÿÿÿÿÿÿÿÿÿ§
F__inference_dense_1_layer_call_and_return_conditional_losses_442077764]/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_dense_1_layer_call_fn_442077746P/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ³
F__inference_model_1_layer_call_and_return_conditional_losses_442077514i78 8¢5
.¢+
!
input_2ÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ³
F__inference_model_1_layer_call_and_return_conditional_losses_442077553i78 8¢5
.¢+
!
input_2ÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ²
F__inference_model_1_layer_call_and_return_conditional_losses_442077670h78 7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ²
F__inference_model_1_layer_call_and_return_conditional_losses_442077709h78 7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_model_1_layer_call_fn_442077203\78 8¢5
.¢+
!
input_2ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_model_1_layer_call_fn_442077475\78 8¢5
.¢+
!
input_2ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_model_1_layer_call_fn_442077592[78 7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_model_1_layer_call_fn_442077631[78 7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ«
J__inference_predictions_layer_call_and_return_conditional_losses_442077800] 0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
/__inference_predictions_layer_call_fn_442077782P 0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¬
'__inference_signature_wrapper_44207772878 ;¢8
¢ 
1ª.
,
input_2!
input_2ÿÿÿÿÿÿÿÿÿ"9ª6
4
predictions%"
predictionsÿÿÿÿÿÿÿÿÿ