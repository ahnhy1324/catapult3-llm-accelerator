module bitnet_tl5 #(
    parameter int LANES = 10,
    parameter int ACT_W = 8,
    parameter int OUT_W = 24,
    parameter int REDUCTION_GROUP = 8,
    parameter int PIPE_DEPTH = 2,
    localparam int GROUPS = (LANES + 4) / 5,
    localparam int REDUCTION_WORDS = (GROUPS + REDUCTION_GROUP - 1) / REDUCTION_GROUP
) (
    input  logic clk,
    input  logic rst_n,
    input  logic build_start,
    input  logic signed [ACT_W-1:0] activation [0:LANES-1],
    output logic table_ready,
    input  logic in_valid,
    input  logic [7:0] packed_weight [0:GROUPS-1],
    output logic out_valid,
    output logic signed [OUT_W-1:0] out_value,
    output logic saturation
);
    initial begin
        if (LANES < 1 || REDUCTION_GROUP < 1 || PIPE_DEPTH < 1) $fatal(1, "invalid parameters");
    end

    function automatic longint signed sat_width(input longint signed value, input int width);
        longint signed maximum;
        longint signed minimum;
        begin
            maximum = (64'sd1 <<< (width - 1)) - 1;
            minimum = -(64'sd1 <<< (width - 1));
            if (value > maximum) sat_width = maximum;
            else if (value < minimum) sat_width = minimum;
            else sat_width = value;
        end
    endfunction

    logic signed [ACT_W-1:0] activation_build_reg [0:LANES-1];
    function automatic logic signed [OUT_W-1:0] entry_value(
        input int group_number,
        input logic [1:0] digit0,
        input logic [1:0] digit1,
        input logic [1:0] digit2,
        input logic [1:0] digit3,
        input logic [1:0] digit4
    );
        logic signed [OUT_W-1:0] term0;
        logic signed [OUT_W-1:0] term1;
        logic signed [OUT_W-1:0] term2;
        logic signed [OUT_W-1:0] term3;
        logic signed [OUT_W-1:0] term4;
        begin
            term0 = '0;
            term1 = '0;
            term2 = '0;
            term3 = '0;
            term4 = '0;
            if (group_number * 5 < LANES) begin
                term0 = $signed(activation_build_reg[group_number * 5]);
                if (digit0 == 0) term0 = -term0;
                else if (digit0 == 1) term0 = '0;
            end
            if (group_number * 5 + 1 < LANES) begin
                term1 = $signed(activation_build_reg[group_number * 5 + 1]);
                if (digit1 == 0) term1 = -term1;
                else if (digit1 == 1) term1 = '0;
            end
            if (group_number * 5 + 2 < LANES) begin
                term2 = $signed(activation_build_reg[group_number * 5 + 2]);
                if (digit2 == 0) term2 = -term2;
                else if (digit2 == 1) term2 = '0;
            end
            if (group_number * 5 + 3 < LANES) begin
                term3 = $signed(activation_build_reg[group_number * 5 + 3]);
                if (digit3 == 0) term3 = -term3;
                else if (digit3 == 1) term3 = '0;
            end
            if (group_number * 5 + 4 < LANES) begin
                term4 = $signed(activation_build_reg[group_number * 5 + 4]);
                if (digit4 == 0) term4 = -term4;
                else if (digit4 == 1) term4 = '0;
            end
            entry_value = (term0 + term1) + (term2 + term3) + term4;
        end
    endfunction

    logic signed [OUT_W-1:0] lut_table [0:GROUPS-1][0:242];
    logic build_active;
    // Each RAM bank keeps its own binary write address and base-3 odometer.
    // The Quartus attributes prevent equivalent-state merging from recreating
    // one high-fan-out address/decode cone across every bank.
    (* preserve, dont_merge *) logic [7:0] build_address_bank [0:GROUPS-1];
    (* preserve, dont_merge *) logic [1:0] build_digit_bank [0:GROUPS-1][0:4];
    logic signed [OUT_W-1:0] build_value_reg [0:GROUPS-1];
    logic [7:0] build_write_address_reg [0:GROUPS-1];
    logic build_value_valid;
    logic build_value_last;
    integer build_group;
    integer build_input_index;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            build_active <= 1'b0;
            build_value_valid <= 1'b0;
            build_value_last <= 1'b0;
            table_ready <= 1'b0;
        end else if (build_start) begin
            for (build_input_index = 0; build_input_index < LANES; build_input_index = build_input_index + 1)
                activation_build_reg[build_input_index] <= activation[build_input_index];
            build_active <= 1'b1;
            for (build_input_index = 0; build_input_index < GROUPS; build_input_index = build_input_index + 1) begin
                build_address_bank[build_input_index] <= '0;
                build_digit_bank[build_input_index][0] <= '0;
                build_digit_bank[build_input_index][1] <= '0;
                build_digit_bank[build_input_index][2] <= '0;
                build_digit_bank[build_input_index][3] <= '0;
                build_digit_bank[build_input_index][4] <= '0;
            end
            build_value_valid <= 1'b0;
            build_value_last <= 1'b0;
            table_ready <= 1'b0;
        end else begin
            // The RAM write consumes values registered one cycle earlier, so
            // the table-building arithmetic is not in the RAM input path.
            if (build_value_valid) begin
                for (build_group = 0; build_group < GROUPS; build_group = build_group + 1)
                    lut_table[build_group][build_write_address_reg[build_group]] <= build_value_reg[build_group];
                if (build_value_last) table_ready <= 1'b1;
            end

            build_value_valid <= build_active;
            build_value_last <= build_active && build_address_bank[0] == 8'd242;
            if (build_active) begin
                for (build_group = 0; build_group < GROUPS; build_group = build_group + 1) begin
                    build_value_reg[build_group] <= entry_value(
                        build_group,
                        build_digit_bank[build_group][0],
                        build_digit_bank[build_group][1],
                        build_digit_bank[build_group][2],
                        build_digit_bank[build_group][3],
                        build_digit_bank[build_group][4]
                    );
                    build_write_address_reg[build_group] <= build_address_bank[build_group];
                    build_address_bank[build_group] <= build_address_bank[build_group] + 1'b1;
                    if (build_digit_bank[build_group][0] != 2) begin
                        build_digit_bank[build_group][0] <= build_digit_bank[build_group][0] + 1'b1;
                    end else begin
                        build_digit_bank[build_group][0] <= '0;
                        if (build_digit_bank[build_group][1] != 2) begin
                            build_digit_bank[build_group][1] <= build_digit_bank[build_group][1] + 1'b1;
                        end else begin
                            build_digit_bank[build_group][1] <= '0;
                            if (build_digit_bank[build_group][2] != 2) begin
                                build_digit_bank[build_group][2] <= build_digit_bank[build_group][2] + 1'b1;
                            end else begin
                                build_digit_bank[build_group][2] <= '0;
                                if (build_digit_bank[build_group][3] != 2) begin
                                    build_digit_bank[build_group][3] <= build_digit_bank[build_group][3] + 1'b1;
                                end else begin
                                    build_digit_bank[build_group][3] <= '0;
                                    if (build_digit_bank[build_group][4] != 2)
                                        build_digit_bank[build_group][4] <= build_digit_bank[build_group][4] + 1'b1;
                                    else build_digit_bank[build_group][4] <= '0;
                                end
                            end
                        end
                    end
                end
                if (build_address_bank[0] == 8'd242) build_active <= 1'b0;
            end
        end
    end

    logic input_valid_reg;
    logic [7:0] packed_weight_reg [0:GROUPS-1];
    integer input_index;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) input_valid_reg <= 1'b0;
        else begin
            input_valid_reg <= in_valid && table_ready;
            if (in_valid && table_ready)
                for (input_index = 0; input_index < GROUPS; input_index = input_index + 1)
                    packed_weight_reg[input_index] <= packed_weight[input_index];
        end
    end

    logic signed [OUT_W-1:0] group_value_reg [0:GROUPS-1];
    logic group_invalid_reg [0:GROUPS-1];
    logic valid_stage1;
    integer read_group;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) valid_stage1 <= 1'b0;
        else begin
            valid_stage1 <= input_valid_reg;
            for (read_group = 0; read_group < GROUPS; read_group = read_group + 1) begin
                if (packed_weight_reg[read_group] <= 8'd242)
                    group_value_reg[read_group] <= lut_table[read_group][packed_weight_reg[read_group]];
                else group_value_reg[read_group] <= '0;
                group_invalid_reg[read_group] <= packed_weight_reg[read_group] > 8'd242;
            end
        end
    end

    logic signed [OUT_W-1:0] tree_reg [0:7][0:127];
    logic tree_invalid_reg [0:7][0:127];
    logic valid_tree [0:7];
    integer tree0_index;
    integer tree1_index;
    integer tree2_index;
    integer tree3_index;
    integer tree4_index;
    integer tree5_index;
    integer tree6_index;
    integer tree7_index;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_tree[0] <= 1'b0;
            valid_tree[1] <= 1'b0;
            valid_tree[2] <= 1'b0;
            valid_tree[3] <= 1'b0;
            valid_tree[4] <= 1'b0;
            valid_tree[5] <= 1'b0;
            valid_tree[6] <= 1'b0;
            valid_tree[7] <= 1'b0;
        end else begin
            valid_tree[0] <= valid_stage1;
            for (tree0_index = 0; tree0_index < 128; tree0_index = tree0_index + 1) begin
                if (tree0_index * 2 + 1 < GROUPS) begin
                    tree_reg[0][tree0_index] <= $signed(group_value_reg[tree0_index * 2]) + $signed(group_value_reg[tree0_index * 2 + 1]);
                    tree_invalid_reg[0][tree0_index] <= group_invalid_reg[tree0_index * 2] || group_invalid_reg[tree0_index * 2 + 1];
                end else if (tree0_index * 2 < GROUPS) begin
                    tree_reg[0][tree0_index] <= group_value_reg[tree0_index * 2];
                    tree_invalid_reg[0][tree0_index] <= group_invalid_reg[tree0_index * 2];
                end else begin
                    tree_reg[0][tree0_index] <= '0;
                    tree_invalid_reg[0][tree0_index] <= 1'b0;
                end
            end
            valid_tree[1] <= valid_tree[0];
            for (tree1_index = 0; tree1_index < 64; tree1_index = tree1_index + 1) begin
                tree_reg[1][tree1_index] <= $signed(tree_reg[0][tree1_index * 2]) + $signed(tree_reg[0][tree1_index * 2 + 1]);
                tree_invalid_reg[1][tree1_index] <= tree_invalid_reg[0][tree1_index * 2] || tree_invalid_reg[0][tree1_index * 2 + 1];
            end
            valid_tree[2] <= valid_tree[1];
            for (tree2_index = 0; tree2_index < 32; tree2_index = tree2_index + 1) begin
                tree_reg[2][tree2_index] <= $signed(tree_reg[1][tree2_index * 2]) + $signed(tree_reg[1][tree2_index * 2 + 1]);
                tree_invalid_reg[2][tree2_index] <= tree_invalid_reg[1][tree2_index * 2] || tree_invalid_reg[1][tree2_index * 2 + 1];
            end
            valid_tree[3] <= valid_tree[2];
            for (tree3_index = 0; tree3_index < 16; tree3_index = tree3_index + 1) begin
                tree_reg[3][tree3_index] <= $signed(tree_reg[2][tree3_index * 2]) + $signed(tree_reg[2][tree3_index * 2 + 1]);
                tree_invalid_reg[3][tree3_index] <= tree_invalid_reg[2][tree3_index * 2] || tree_invalid_reg[2][tree3_index * 2 + 1];
            end
            valid_tree[4] <= valid_tree[3];
            for (tree4_index = 0; tree4_index < 8; tree4_index = tree4_index + 1) begin
                tree_reg[4][tree4_index] <= $signed(tree_reg[3][tree4_index * 2]) + $signed(tree_reg[3][tree4_index * 2 + 1]);
                tree_invalid_reg[4][tree4_index] <= tree_invalid_reg[3][tree4_index * 2] || tree_invalid_reg[3][tree4_index * 2 + 1];
            end
            valid_tree[5] <= valid_tree[4];
            for (tree5_index = 0; tree5_index < 4; tree5_index = tree5_index + 1) begin
                tree_reg[5][tree5_index] <= $signed(tree_reg[4][tree5_index * 2]) + $signed(tree_reg[4][tree5_index * 2 + 1]);
                tree_invalid_reg[5][tree5_index] <= tree_invalid_reg[4][tree5_index * 2] || tree_invalid_reg[4][tree5_index * 2 + 1];
            end
            valid_tree[6] <= valid_tree[5];
            for (tree6_index = 0; tree6_index < 2; tree6_index = tree6_index + 1) begin
                tree_reg[6][tree6_index] <= $signed(tree_reg[5][tree6_index * 2]) + $signed(tree_reg[5][tree6_index * 2 + 1]);
                tree_invalid_reg[6][tree6_index] <= tree_invalid_reg[5][tree6_index * 2] || tree_invalid_reg[5][tree6_index * 2 + 1];
            end
            valid_tree[7] <= valid_tree[6];
            for (tree7_index = 0; tree7_index < 1; tree7_index = tree7_index + 1) begin
                tree_reg[7][tree7_index] <= $signed(tree_reg[6][tree7_index * 2]) + $signed(tree_reg[6][tree7_index * 2 + 1]);
                tree_invalid_reg[7][tree7_index] <= tree_invalid_reg[6][tree7_index * 2] || tree_invalid_reg[6][tree7_index * 2 + 1];
            end
        end
    end

    logic signed [OUT_W-1:0] value_pipe [0:PIPE_DEPTH-1];
    logic valid_pipe [0:PIPE_DEPTH-1];
    logic saturation_pipe [0:PIPE_DEPTH-1];
    integer pipe_index;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (pipe_index = 0; pipe_index < PIPE_DEPTH; pipe_index = pipe_index + 1) begin
                value_pipe[pipe_index] <= '0;
                valid_pipe[pipe_index] <= 1'b0;
                saturation_pipe[pipe_index] <= 1'b0;
            end
        end else begin
            value_pipe[0] <= tree_reg[7][0];
            valid_pipe[0] <= valid_tree[7];
            saturation_pipe[0] <= tree_invalid_reg[7][0];
            for (pipe_index = 1; pipe_index < PIPE_DEPTH; pipe_index = pipe_index + 1) begin
                value_pipe[pipe_index] <= value_pipe[pipe_index-1];
                valid_pipe[pipe_index] <= valid_pipe[pipe_index-1];
                saturation_pipe[pipe_index] <= saturation_pipe[pipe_index-1];
            end
        end
    end
    assign out_value = value_pipe[PIPE_DEPTH-1];
    assign out_valid = valid_pipe[PIPE_DEPTH-1];
    assign saturation = saturation_pipe[PIPE_DEPTH-1];
endmodule
