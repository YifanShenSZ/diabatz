void test_DqHd();
void test_DcHd();
void test_DcDqHd();

void test_DqHa();
void test_DcHa();
void test_Dc_DqH_a();

void test_composite();
void test_DcHc();
void test_Dc_DqH_c();

int main() {
    test_DqHd();
    test_DcHd();
    test_DcDqHd();

    test_DqHa(); // Only for debugging, will not appear in fitting
    test_DcHa(); // Only for debugging, will not appear in fitting
    // d / dc * (d / dq * H)a, i.e. the gradient of adiabatic d / dq * H matrix element over c
    test_Dc_DqH_a();

    test_composite(); // Only for debugging, will not appear in fitting
    test_DcHc();
    // d / dc * (d / dq * H)c, i.e. the gradient of composite d / dq * H matrix element over c
    test_Dc_DqH_c();
}