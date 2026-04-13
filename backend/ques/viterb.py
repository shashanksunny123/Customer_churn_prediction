print("Viterbi algorithm:")
states=int(input("Enter the number of states: "))
prob={},state_names={}
for i in range(states):
    state_names[i]=input("Enter the name of the state:")
    prob[i]=float(input("Enter the initial probability of the state:"))
out=input("enter the output sequence:").split()
emit={} 
trans={} 
for o in out:
    for i in range(states):
        emit[i,o]=float(input(f"Enter the emission probability for {o} and {i}"))
for j in range(states):
    for i in range(states):
        trans[i,j]=float(input(f"Enter the transition probability from state {i} to state {j}:"))
delta=[list(prob.values())]
path=[[i] for i in range(states)]

for o in out:
    new_delta=[],new_path=[]
    for j in range(states):
        vals=[delta[-1][j] * emit[j,o] * trans[i,j] for j in range(states)]
        max_val=max(vals)
        idx=vals.index(max_val)
        new_delta.append(max_val)
        new_path.append(path[idx] + [i])
    delta.append(new_delta)
    path=new_path
max_prob=max(delta[-1])
best_path=path[delta[-1].index(max_prob)] #indexes only 
print(" Best sequence:", [state_names[i] for i in best_path[i]])