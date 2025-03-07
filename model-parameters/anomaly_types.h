/* Generated by Edge Impulse
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef _EI_CLASSIFIER_ANOMALY_TYPES_HEADER_H_
#define _EI_CLASSIFIER_ANOMALY_TYPES_HEADER_H_

#define EI_CLASSIFIER_HAS_ANOMALY    1

const uint16_t EI_CLASSIFIER_ANOM_AXIS[]  { 0, 1, 7, 11, 12, 18, 22, 23, 29 };
#define EI_CLASSIFIER_ANOM_AXIS_SIZE      9
#define EI_CLASSIFIER_ANOM_CLUSTER_COUNT  32

typedef struct {
float centroid[EI_CLASSIFIER_ANOM_AXIS_SIZE];
float max_error;
} ei_classifier_anom_cluster_t;

#endif // _EI_CLASSIFIER_ANOMALY_TYPES_HEADER_H_
