# PACMAN imports
# from spynnaker.pyNN.models.common.population_settable_change_requires_mapping import \
#     PopulationSettableChangeRequiresMapping

# from spynnaker.pyNN.models.abstract_models import AbstractPopulationSettable
from spinn_front_end_common.abstract_models import AbstractChangableAfterRun

from pacman.executor.injection_decorator import inject_items
from pacman.model.constraints.key_allocator_constraints import ContiguousKeyRangeContraint
from pacman.model.decorators.overrides import overrides
from pacman.model.graphs.application import ApplicationVertex
from pacman.model.resources.cpu_cycles_per_tick_resource import \
    CPUCyclesPerTickResource
from pacman.model.resources.dtcm_resource import DTCMResource
from pacman.model.resources.resource_container import ResourceContainer
from pacman.model.resources.sdram_resource import SDRAMResource

# SpinnFrontEndCommon imports
# from spinn_front_end_common.abstract_models \
#     .abstract_binary_uses_simulation_run import AbstractBinaryUsesSimulationRun
from spinn_front_end_common.abstract_models \
    .abstract_generates_data_specification \
    import AbstractGeneratesDataSpecification
from spinn_front_end_common.abstract_models.abstract_has_associated_binary \
    import AbstractHasAssociatedBinary
from spinn_front_end_common.abstract_models. \
    abstract_provides_outgoing_partition_constraints import \
    AbstractProvidesOutgoingPartitionConstraints
from spinn_front_end_common.utilities import globals_variables

from spinn_front_end_common.interface.simulation import simulation_utilities
from spinn_front_end_common.utilities import constants as \
    front_end_common_constants
from spinn_front_end_common.utilities.utility_objs.executable_start_type \
    import ExecutableStartType

from spinn_front_end_common.utilities import globals_variables

# sPyNNaker imports
from spynnaker.pyNN.models.abstract_models import AbstractAcceptsIncomingSynapses
from spynnaker.pyNN.models.neuron import AbstractPopulationVertex
from spynnaker.pyNN.utilities import constants
from spynnaker.pyNN.models.common.simple_population_settable \
    import SimplePopulationSettable

from data_specification.enums.data_type import DataType

# Breakout imports
from convolution_population_machine_vertex import \
                                        ConvolutionPopulationMachineVertex
from enum import Enum
import os
import numpy

# **HACK** for Projection to connect a synapse type is required
class ConvolutionSynapseType(object):
    def get_synapse_id_by_target(self, target):
        return 0

def subsamp_size(start, end, step):
    return numpy.uint32( ((end - start - 1) // step) + 1)


KEY_FORMATS = Enum(value="INPUT_KEY_FORMATS",
                   names=[('USE_XYP', 0),
                          ('USE_PYX', 1)])

class ConvolutionPopulation(ApplicationVertex, AbstractGeneratesDataSpecification,
               AbstractHasAssociatedBinary, 
               AbstractProvidesOutgoingPartitionConstraints,
               AbstractAcceptsIncomingSynapses,
               SimplePopulationSettable,
               # AbstractBinaryUsesSimulationRun
               ):

    def get_connections_from_machine(self, transceiver, placement, edge, graph_mapper, 
                               routing_infos, synapse_information, machine_time_step):
        
        super(ConvolutionPopulation, self).get_connections_from_machine(
                            transceiver, placement, edge, graph_mapper,
                            routing_infos, synapse_information, machine_time_step)

    def set_synapse_dynamics(self, synapse_dynamics):
        pass

    def add_pre_run_connection_holder(self, connection_holder, projection_edge,
                                      synapse_information):

        super(ConvolutionPopulation, self).add_pre_run_connection_holder(
                        connection_holder, projection_edge, synapse_information)

    # def get_binary_start_type(self):
    #     super(Breakout, self).get_binary_start_type()
    #
    # def requires_mapping(self):
    #     pass

    def clear_connection_cache(self):
        pass

    WIDTH_PIXELS = 160
    HEIGHT_PIXELS = 128
    POLARITY = 0
    POLARITY_BITS = 1
    SAMPLE_STEP_WIDTH = 1
    SAMPLE_STEP_HEIGHT = 1
    KERNEL = numpy.array( #centre-surround
       [[-0.12771209, -0.08770193, -0.12771209],
       [-0.12771209,  0.94167642, -0.12771209],
       [-0.12771209, -0.08770193, -0.12771209]], dtype='float16')
    THRESHOLD = 1.
    TIME_WINDOW = 3
    # **HACK** for Projection to connect a synapse type is required
    synapse_type = ConvolutionSynapseType()

    def __init__(self, n_neurons,
                 spikes_per_second=AbstractPopulationVertex.
                 none_pynn_default_parameters['spikes_per_second'],
                 ring_buffer_sigma=AbstractPopulationVertex.
                 none_pynn_default_parameters['ring_buffer_sigma'],
                 incoming_spike_buffer_size=None,
                 constraints=None,
                 label="Convolution core",
                 src_width=WIDTH_PIXELS, src_height=HEIGHT_PIXELS,
                 src_polarity_bits=POLARITY_BITS,
                 polarity=POLARITY,
                 sample_step_width=SAMPLE_STEP_WIDTH,
                 sample_step_height=SAMPLE_STEP_HEIGHT,
                 kernel=KERNEL,
                 threshold=THRESHOLD,
                 use_xyp_or_pyx=KEY_FORMATS.USE_XYP.value,
                 time_window=TIME_WINDOW
                 ):
        # **NOTE** n_neurons currently ignored - width and height will be
        # specified as additional parameters, forcing their product to be
        # duplicated in n_neurons seems pointless

        self._width = numpy.uint32(src_width) #1
        self._height = numpy.uint32(src_height) #2
        self._polarity_bits = numpy.uint32(src_polarity_bits) #3
        self._width_bits = numpy.uint32(numpy.ceil(numpy.log2(src_width))) #4
        self._height_bits = numpy.uint32(numpy.ceil(numpy.log2(src_height))) #5

        self._kernel_width = numpy.uint32(kernel.shape[1]) #6
        self._kernel_height = numpy.uint32(kernel.shape[0]) #7

        self._step_width = numpy.uint32(numpy.round(numpy.log2(sample_step_width))) #8
        self._step_height = numpy.uint32(numpy.round(numpy.log2(sample_step_height))) #9

        self._threshold = numpy.float16(threshold) #10
        self._use_xyp_or_pyx = numpy.uint32(use_xyp_or_pyx)#11
        self._time_window = numpy.uint32(time_window) #12

        self._start_width = kernel.shape[1]//2
        self._start_height = kernel.shape[0]//2

        self._out_width = subsamp_size(self._start_width, self._width,
                                       1<<self._step_width) #13
        self._out_height = subsamp_size(self._start_height, self._height,
                                        1<<self._step_height) #14

        self._out_width_bits = numpy.uint32(
                                    numpy.ceil(numpy.log2(self._out_width))) #15
        self._out_height_bits = numpy.uint32(
                                    numpy.ceil(numpy.log2(self._out_height))) #16

        self._polarity = polarity #17

        self._n_neurons = (1 << (self._out_width_bits + self._out_height_bits +
                                 self._polarity_bits))

        self._kernel = numpy.float16(kernel) # M

        #params, kernel, key
        self._memory_size_in_bytes = (17 + kernel.size + 1)*4



        # Superclasses
        ApplicationVertex.__init__(
            self, label, constraints, self.n_atoms)
        AbstractProvidesOutgoingPartitionConstraints.__init__(self)
        SimplePopulationSettable.__init__(self)
        AbstractChangableAfterRun.__init__(self)
        AbstractAcceptsIncomingSynapses.__init__(self)
        self._change_requires_mapping = True
        # get config from simulator
        config = globals_variables.get_simulator().config

        if incoming_spike_buffer_size is None:
            self._incoming_spike_buffer_size = config.getint(
                                    "Simulation", "incoming_spike_buffer_size")

        # PopulationSettableChangeRequiresMapping.__init__(self)
        # self.width = width
        # self.height = height

    def get_maximum_delay_supported_in_ms(self, machine_time_step):
        # Breakout has no synapses so can simulate only one time step of delay
        return machine_time_step / 1000.0

#    def get_max_atoms_per_core(self):
 #       return self.n_atoms

    # ------------------------------------------------------------------------
    # ApplicationVertex overrides
    # ------------------------------------------------------------------------
    @overrides(ApplicationVertex.get_resources_used_by_atoms)
    def get_resources_used_by_atoms(self, vertex_slice):
        # **HACK** only way to force no partitioning is to zero dtcm and cpu
        container = ResourceContainer(
            sdram=SDRAMResource(
                self._memory_size_in_bytes +
                front_end_common_constants.SYSTEM_BYTES_REQUIREMENT),
            dtcm=DTCMResource(0),
            cpu_cycles=CPUCyclesPerTickResource(0))

        return container

    @overrides(ApplicationVertex.create_machine_vertex)
    def create_machine_vertex(self, vertex_slice, resources_required,
                              label=None, constraints=None):
        # Return suitable machine vertex
        return ConvolutionPopulationMachineVertex(resources_required, constraints, label)

    @property
    @overrides(ApplicationVertex.n_atoms)
    def n_atoms(self):

        # **TODO** should we calculate this automatically
        # based on log2 of width and height?
        return self._n_neurons

    # ------------------------------------------------------------------------
    # AbstractGeneratesDataSpecification overrides
    # ------------------------------------------------------------------------
    @inject_items({"machine_time_step": "MachineTimeStep",
                   "time_scale_factor": "TimeScaleFactor",
                   "graph_mapper": "MemoryGraphMapper",
                   "routing_info": "MemoryRoutingInfos",
                   "tags": "MemoryTags",
                   "n_machine_time_steps": "TotalMachineTimeSteps"})
    @overrides(AbstractGeneratesDataSpecification.generate_data_specification,
               additional_arguments={"machine_time_step", "time_scale_factor",
                                     "graph_mapper", "routing_info", "tags",
                                     "n_machine_time_steps"}
               )
    def generate_data_specification(self, spec, placement, machine_time_step,
                                    time_scale_factor, graph_mapper,
                                    routing_info, tags, n_machine_time_steps):
        vertex = placement.vertex
        vertex_slice = graph_mapper.get_slice(vertex)

        spec.comment("\n*** Spec for Breakout Instance ***\n\n")
        spec.comment("\nReserving memory space for data regions:\n\n")

        # Reserve memory:
        spec.reserve_memory_region(
            region=ConvolutionPopulationMachineVertex._MEMORY_REGIONS.SYSTEM.value,
            size=front_end_common_constants.SYSTEM_BYTES_REQUIREMENT,
            label='setup')

        spec.reserve_memory_region(
            region=ConvolutionPopulationMachineVertex.
                                _MEMORY_REGIONS.CONVOLUTION.value,
            size=self._memory_size_in_bytes, label='Convolution Parameters')

        # vertex.reserve_provenance_data_region(spec)

        # Write setup region
        spec.comment("\nWriting Setup region:\n")
        spec.switch_write_focus(
            ConvolutionPopulationMachineVertex._MEMORY_REGIONS.SYSTEM.value)
        spec.write_array(simulation_utilities.get_simulation_header_array(
            self.get_binary_file_name(), machine_time_step,
            time_scale_factor))

        # Write breakout region containing routing key to transmit with
        spec.comment("\nWriting Convolution region:\n")
        spec.switch_write_focus(
            ConvolutionPopulationMachineVertex._MEMORY_REGIONS.CONVOLUTION.value)

        routing_key = routing_info.get_first_key_from_pre_vertex(
                                        vertex, constants.SPIKE_PARTITION_ID)
        if routing_key is None:
            routing_key = 0
        # print(vertex, routing_key)

        spec.write_value(routing_key, data_type=DataType.UINT32)

        spec.write_value(self._width, data_type=DataType.UINT32)
        spec.write_value(self._height, data_type=DataType.UINT32)
        spec.write_value(self._out_width, data_type=DataType.UINT32)
        spec.write_value(self._out_height, data_type=DataType.UINT32)
        spec.write_value(self._kernel_width, data_type=DataType.UINT32)
        spec.write_value(self._kernel_height, data_type=DataType.UINT32)
        spec.write_value(self._step_width, data_type=DataType.UINT32)
        spec.write_value(self._step_height, data_type=DataType.UINT32)

        spec.write_value(self._width_bits, data_type=DataType.UINT32)
        spec.write_value(self._height_bits, data_type=DataType.UINT32)
        spec.write_value(self._polarity_bits, data_type=DataType.UINT32)
        spec.write_value(self._out_width_bits, data_type=DataType.UINT32)
        spec.write_value(self._out_height_bits, data_type=DataType.UINT32)

        spec.write_value(float(self._threshold), data_type=DataType.S1615)
        spec.write_value(self._use_xyp_or_pyx, data_type=DataType.UINT32)
        spec.write_value(self._time_window, data_type=DataType.UINT32)
        spec.write_value(self._polarity, data_type=DataType.UINT32)

        for r in range(self._kernel.shape[0]):
            for c in range(self._kernel.shape[1]):
                spec.write_value(
                        float(self._kernel[r,c]), data_type=DataType.S1615)

        # End-of-Spec:
        spec.end_specification()

    # ------------------------------------------------------------------------
    # AbstractHasAssociatedBinary overrides
    # ------------------------------------------------------------------------
    @overrides(AbstractHasAssociatedBinary.get_binary_file_name)
    def get_binary_file_name(self):
        return "convolution_core.aplx"
        # return os.path.join(os.path.dirname(__file__),
        #                     "../model_binaries", "convolution_core.aplx")

    @overrides(AbstractHasAssociatedBinary.get_binary_start_type)
    def get_binary_start_type(self):
        return ExecutableStartType.USES_SIMULATION_INTERFACE

    # ------------------------------------------------------------------------
    # AbstractProvidesOutgoingPartitionConstraints overrides
    # ------------------------------------------------------------------------
    @overrides(AbstractProvidesOutgoingPartitionConstraints.
               get_outgoing_partition_constraints)
    def get_outgoing_partition_constraints(self, partition):
        return [ContiguousKeyRangeContraint()]

    @property
    @overrides(AbstractChangableAfterRun.requires_mapping)
    def requires_mapping(self):
        return self._change_requires_mapping

    @overrides(AbstractChangableAfterRun.mark_no_changes)
    def mark_no_changes(self):
        self._change_requires_mapping = False

    @overrides(SimplePopulationSettable.set_value)
    def set_value(self, key, value):
        SimplePopulationSettable.set_value(self, key, value)
        self._change_requires_neuron_parameters_reload = True
