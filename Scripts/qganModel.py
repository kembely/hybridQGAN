import torch
import torch.nn as nn
import pennylane as qml

# Include a dense classical layer to upscale z values to number of features required by the discriminator
class HQuantumGenerator(nn.Module):
    """Quantum generator class for the patch method"""

    def __init__(self, device, num_features, n_qubits=4, n_layers=1):
        """
        Args:
            device (torch.device or str): Target execution device for tensor computations and circuit simulation (e.g., 'cpu' or 'cuda').
            num_features (int): Number of input traffic features (classical feature dimension) to be mapped or generated.
            n_qubits (int, optional): Number of qubits used in the parameterized quantum circuit (PQC) ansatz. Defaults to 4.
            n_layers (int, optional): Number of repeated variational layers (entanglement and parameterized rotation gates). Defaults to 1.
        """
        super().__init__()
        self.device=device
        self.features= num_features
        # Quantum variables
        self.n_qubits = n_qubits  
        self.q_depth = n_layers  
        self.fc_outputLayer = nn.Linear(self.n_qubits, self.features)
        self.tanh = nn.Tanh() # data is normalized around -1 y 1 .. sigmoid if normalized around 0 and 1 
        self.q_params = nn.init.xavier_normal_(nn.Parameter(torch.rand(self.q_depth, self.n_qubits), requires_grad=True)) 
        # Quantum simulator
        self.q_device = qml.device("lightning.qubit", wires=self.n_qubits)
        self.qnode = qml.QNode(self.quantum_circuit, self.q_device)
        #self.qnode = qml.QNode(self.quantum_circuit, self.q_device, interface="torch", diff_method="parameter-shift") #if parameter-shift is needed
 
    def forward(self, inputs):
    # Create a Tensor to 'catch' a batch of images from the for loop. x.size(0) is the batch size.
        images = []
        for elem in inputs:
            q_out = torch.stack(self.qnode(elem, self.q_params)).unsqueeze(0)
            images.append(q_out)
        images = torch.cat(images,dim=0)
        output  = self.tanh(self.fc_outputLayer(images))
        return output 
       
    def quantum_circuit(self, inputs, weights):    
        # Initialise latent vectors
        for i in range(self.n_qubits):
            qml.RX(inputs[i], wires=i)
        #qml.AngleEmbedding(inputs, wires=range(self.n_qubits), rotation='X')
        # Repeated layer
        for i in range(self.q_depth):
            # Parameterised layer
            for y in range(self.n_qubits):
                qml.RY(weights[i][y], wires=y)
            # Control CNOT gates
            for y in range(self.n_qubits - 1):
                qml.CNOT(wires=[y, y + 1])

        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

#CICDDoS2019 dataset
class fully_Classical_Discriminator2019(nn.Module):
    """Fully connected classical discriminator"""

    def __init__(self, num_features):
        super().__init__()
        self.features = num_features
        self.dim=self.features//2 
        self.fc1= nn.Linear(self.features, self.dim)  
        self.fc2= nn.Linear(self.dim, self.dim//2)     
        self.fc3= nn.Linear(self.dim//2, 8 ) 
        self.fc_outputLayer1 = nn.Linear(8, 1) 
        self.lRelu= nn.LeakyReLU() 
       
    def forward(self, x):
        output = self.lRelu(self.fc1(x))
        output = self.lRelu(self.fc2(output))
        output = self.lRelu(self.fc3(output))
        output = self.fc_outputLayer1(output)
        return output
    
#CICIoMT2024 dataset
class fully_Classical_Discriminator2024(nn.Module):
    """Fully connected classical discriminator"""

    def __init__(self, num_features):
        super().__init__()
        self.features = num_features
        self.dim = 34 
        self.fc1= nn.Linear(self.features, self.dim) 
        self.fc2= nn.Linear(self.dim, 10)     
        self.fc3= nn.Linear(10, 8) 
        self.fc_outputLayer1 = nn.Linear(8, 1) 

        self.lRelu= nn.ReLU()  

    def forward(self, x):
        output = self.lRelu(self.fc1(x))
        output = self.lRelu(self.fc2(output))
        output = self.lRelu(self.fc3(output))
        output = self.fc_outputLayer1(output)
        return output
