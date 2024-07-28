/*
 * Interface wrapper code.
 *
 * Generated by SIP 4.19.3
 *
 * Copyright (C) 2009 Autodesk, Inc. and/or its licensors.
 * All Rights Reserved.
 * 
 * The coded instructions, statements, computer programs, and/or related material 
 * (collectively the "Data") in these files contain unpublished information 
 * proprietary to Autodesk, Inc. and/or its licensors, which is protected by 
 * Canada and United States of America federal copyright law and by international 
 * treaties. 
 * 
 * The Data may not be disclosed or distributed to third parties, in whole or in
 * part, without the prior written consent of Autodesk, Inc. ("Autodesk").
 * 
 * THE DATA IS PROVIDED "AS IS" AND WITHOUT WARRANTY.
 * ALL WARRANTIES ARE EXPRESSLY EXCLUDED AND DISCLAIMED. AUTODESK MAKES NO
 * WARRANTY OF ANY KIND WITH RESPECT TO THE DATA, EXPRESS, IMPLIED OR ARISING
 * BY CUSTOM OR TRADE USAGE, AND DISCLAIMS ANY IMPLIED WARRANTIES OF TITLE, 
 * NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE OR USE. 
 * WITHOUT LIMITING THE FOREGOING, AUTODESK DOES NOT WARRANT THAT THE OPERATION
 * OF THE DATA WILL BE UNINTERRUPTED OR ERROR FREE. 
 * 
 * IN NO EVENT SHALL AUTODESK, ITS AFFILIATES, PARENT COMPANIES, LICENSORS
 * OR SUPPLIERS ("AUTODESK GROUP") BE LIABLE FOR ANY LOSSES, DAMAGES OR EXPENSES
 * OF ANY KIND (INCLUDING WITHOUT LIMITATION PUNITIVE OR MULTIPLE DAMAGES OR OTHER
 * SPECIAL, DIRECT, INDIRECT, EXEMPLARY, INCIDENTAL, LOSS OF PROFITS, REVENUE
 * OR DATA, COST OF COVER OR CONSEQUENTIAL LOSSES OR DAMAGES OF ANY KIND),
 * HOWEVER CAUSED, AND REGARDLESS OF THE THEORY OF LIABILITY, WHETHER DERIVED
 * FROM CONTRACT, TORT (INCLUDING, BUT NOT LIMITED TO, NEGLIGENCE), OR OTHERWISE,
 * ARISING OUT OF OR RELATING TO THE DATA OR ITS USE OR ANY OTHER PERFORMANCE,
 * WHETHER OR NOT AUTODESK HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH LOSS
 * OR DAMAGE. 
 */

#include "sipAPIfbx.h"




PyDoc_STRVAR(doc_FbxWeightedMapping_Reset, "Reset(self, int, int)");

extern "C" {static PyObject *meth_FbxWeightedMapping_Reset(PyObject *, PyObject *);}
static PyObject *meth_FbxWeightedMapping_Reset(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        int a0;
        int a1;
         ::FbxWeightedMapping *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "Bii", &sipSelf, sipType_FbxWeightedMapping, &sipCpp, &a0, &a1))
        {
            sipCpp->Reset(a0,a1);

            Py_INCREF(Py_None);
            return Py_None;
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxWeightedMapping, sipName_Reset, doc_FbxWeightedMapping_Reset);

    return NULL;
}


PyDoc_STRVAR(doc_FbxWeightedMapping_Add, "Add(self, int, int, float)");

extern "C" {static PyObject *meth_FbxWeightedMapping_Add(PyObject *, PyObject *);}
static PyObject *meth_FbxWeightedMapping_Add(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        int a0;
        int a1;
        double a2;
         ::FbxWeightedMapping *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "Biid", &sipSelf, sipType_FbxWeightedMapping, &sipCpp, &a0, &a1, &a2))
        {
            sipCpp->Add(a0,a1,a2);

            Py_INCREF(Py_None);
            return Py_None;
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxWeightedMapping, sipName_Add, doc_FbxWeightedMapping_Add);

    return NULL;
}


PyDoc_STRVAR(doc_FbxWeightedMapping_GetElementCount, "GetElementCount(self, FbxWeightedMapping.ESet) -> int");

extern "C" {static PyObject *meth_FbxWeightedMapping_GetElementCount(PyObject *, PyObject *);}
static PyObject *meth_FbxWeightedMapping_GetElementCount(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxWeightedMapping::ESet a0;
        const  ::FbxWeightedMapping *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BE", &sipSelf, sipType_FbxWeightedMapping, &sipCpp, sipType_FbxWeightedMapping_ESet, &a0))
        {
            int sipRes;

            sipRes = sipCpp->GetElementCount(a0);

            return SIPLong_FromLong(sipRes);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxWeightedMapping, sipName_GetElementCount, doc_FbxWeightedMapping_GetElementCount);

    return NULL;
}


PyDoc_STRVAR(doc_FbxWeightedMapping_GetRelationCount, "GetRelationCount(self, FbxWeightedMapping.ESet, int) -> int");

extern "C" {static PyObject *meth_FbxWeightedMapping_GetRelationCount(PyObject *, PyObject *);}
static PyObject *meth_FbxWeightedMapping_GetRelationCount(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxWeightedMapping::ESet a0;
        int a1;
        const  ::FbxWeightedMapping *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BEi", &sipSelf, sipType_FbxWeightedMapping, &sipCpp, sipType_FbxWeightedMapping_ESet, &a0, &a1))
        {
            int sipRes;

            sipRes = sipCpp->GetRelationCount(a0,a1);

            return SIPLong_FromLong(sipRes);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxWeightedMapping, sipName_GetRelationCount, doc_FbxWeightedMapping_GetRelationCount);

    return NULL;
}


PyDoc_STRVAR(doc_FbxWeightedMapping_GetRelation, "GetRelation(self, FbxWeightedMapping.ESet, int, int) -> FbxWeightedMapping.Element");

extern "C" {static PyObject *meth_FbxWeightedMapping_GetRelation(PyObject *, PyObject *);}
static PyObject *meth_FbxWeightedMapping_GetRelation(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxWeightedMapping::ESet a0;
        int a1;
        int a2;
         ::FbxWeightedMapping *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BEii", &sipSelf, sipType_FbxWeightedMapping, &sipCpp, sipType_FbxWeightedMapping_ESet, &a0, &a1, &a2))
        {
             ::FbxWeightedMapping::Element*sipRes;

            sipRes = &sipCpp->GetRelation(a0,a1,a2);

            return sipConvertFromType(sipRes,sipType_FbxWeightedMapping_Element,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxWeightedMapping, sipName_GetRelation, doc_FbxWeightedMapping_GetRelation);

    return NULL;
}


PyDoc_STRVAR(doc_FbxWeightedMapping_GetRelationIndex, "GetRelationIndex(self, FbxWeightedMapping.ESet, int, int) -> int");

extern "C" {static PyObject *meth_FbxWeightedMapping_GetRelationIndex(PyObject *, PyObject *);}
static PyObject *meth_FbxWeightedMapping_GetRelationIndex(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxWeightedMapping::ESet a0;
        int a1;
        int a2;
        const  ::FbxWeightedMapping *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BEii", &sipSelf, sipType_FbxWeightedMapping, &sipCpp, sipType_FbxWeightedMapping_ESet, &a0, &a1, &a2))
        {
            int sipRes;

            sipRes = sipCpp->GetRelationIndex(a0,a1,a2);

            return SIPLong_FromLong(sipRes);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxWeightedMapping, sipName_GetRelationIndex, doc_FbxWeightedMapping_GetRelationIndex);

    return NULL;
}


PyDoc_STRVAR(doc_FbxWeightedMapping_GetRelationSum, "GetRelationSum(self, FbxWeightedMapping.ESet, int, bool) -> float");

extern "C" {static PyObject *meth_FbxWeightedMapping_GetRelationSum(PyObject *, PyObject *);}
static PyObject *meth_FbxWeightedMapping_GetRelationSum(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxWeightedMapping::ESet a0;
        int a1;
        bool a2;
        const  ::FbxWeightedMapping *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BEib", &sipSelf, sipType_FbxWeightedMapping, &sipCpp, sipType_FbxWeightedMapping_ESet, &a0, &a1, &a2))
        {
            double sipRes;

            sipRes = sipCpp->GetRelationSum(a0,a1,a2);

            return PyFloat_FromDouble(sipRes);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxWeightedMapping, sipName_GetRelationSum, doc_FbxWeightedMapping_GetRelationSum);

    return NULL;
}


PyDoc_STRVAR(doc_FbxWeightedMapping_Normalize, "Normalize(self, FbxWeightedMapping.ESet, bool)");

extern "C" {static PyObject *meth_FbxWeightedMapping_Normalize(PyObject *, PyObject *);}
static PyObject *meth_FbxWeightedMapping_Normalize(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxWeightedMapping::ESet a0;
        bool a1;
         ::FbxWeightedMapping *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BEb", &sipSelf, sipType_FbxWeightedMapping, &sipCpp, sipType_FbxWeightedMapping_ESet, &a0, &a1))
        {
            sipCpp->Normalize(a0,a1);

            Py_INCREF(Py_None);
            return Py_None;
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxWeightedMapping, sipName_Normalize, doc_FbxWeightedMapping_Normalize);

    return NULL;
}


/* Call the instance's destructor. */
extern "C" {static void release_FbxWeightedMapping(void *, int);}
static void release_FbxWeightedMapping(void *sipCppV, int)
{
    delete reinterpret_cast< ::FbxWeightedMapping *>(sipCppV);
}


extern "C" {static void dealloc_FbxWeightedMapping(sipSimpleWrapper *);}
static void dealloc_FbxWeightedMapping(sipSimpleWrapper *sipSelf)
{
    if (sipIsOwnedByPython(sipSelf))
    {
        release_FbxWeightedMapping(sipGetAddress(sipSelf), 0);
    }
}


extern "C" {static void *init_type_FbxWeightedMapping(sipSimpleWrapper *, PyObject *, PyObject *, PyObject **, PyObject **, PyObject **);}
static void *init_type_FbxWeightedMapping(sipSimpleWrapper *, PyObject *sipArgs, PyObject *sipKwds, PyObject **sipUnused, PyObject **, PyObject **sipParseErr)
{
     ::FbxWeightedMapping *sipCpp = 0;

    {
        int a0;
        int a1;

        if (sipParseKwdArgs(sipParseErr, sipArgs, sipKwds, NULL, sipUnused, "ii", &a0, &a1))
        {
            sipCpp = new  ::FbxWeightedMapping(a0,a1);

            return sipCpp;
        }
    }

    {
        const  ::FbxWeightedMapping* a0;

        if (sipParseKwdArgs(sipParseErr, sipArgs, sipKwds, NULL, sipUnused, "J9", sipType_FbxWeightedMapping, &a0))
        {
            sipCpp = new  ::FbxWeightedMapping(*a0);

            return sipCpp;
        }
    }

    return NULL;
}


static PyMethodDef methods_FbxWeightedMapping[] = {
    {SIP_MLNAME_CAST(sipName_Add), meth_FbxWeightedMapping_Add, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxWeightedMapping_Add)},
    {SIP_MLNAME_CAST(sipName_GetElementCount), meth_FbxWeightedMapping_GetElementCount, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxWeightedMapping_GetElementCount)},
    {SIP_MLNAME_CAST(sipName_GetRelation), meth_FbxWeightedMapping_GetRelation, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxWeightedMapping_GetRelation)},
    {SIP_MLNAME_CAST(sipName_GetRelationCount), meth_FbxWeightedMapping_GetRelationCount, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxWeightedMapping_GetRelationCount)},
    {SIP_MLNAME_CAST(sipName_GetRelationIndex), meth_FbxWeightedMapping_GetRelationIndex, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxWeightedMapping_GetRelationIndex)},
    {SIP_MLNAME_CAST(sipName_GetRelationSum), meth_FbxWeightedMapping_GetRelationSum, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxWeightedMapping_GetRelationSum)},
    {SIP_MLNAME_CAST(sipName_Normalize), meth_FbxWeightedMapping_Normalize, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxWeightedMapping_Normalize)},
    {SIP_MLNAME_CAST(sipName_Reset), meth_FbxWeightedMapping_Reset, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxWeightedMapping_Reset)}
};

static sipEnumMemberDef enummembers_FbxWeightedMapping[] = {
    {sipName_eDestination, static_cast<int>( ::FbxWeightedMapping::eDestination), 392},
    {sipName_eSource, static_cast<int>( ::FbxWeightedMapping::eSource), 392},
};

PyDoc_STRVAR(doc_FbxWeightedMapping, "\1FbxWeightedMapping(int, int)\n"
    "FbxWeightedMapping(FbxWeightedMapping)");


sipClassTypeDef sipTypeDef_fbx_FbxWeightedMapping = {
    {
        -1,
        0,
        0,
        SIP_TYPE_CLASS,
        sipNameNr_FbxWeightedMapping,
        {0},
        0
    },
    {
        sipNameNr_FbxWeightedMapping,
        {0, 0, 1},
        8, methods_FbxWeightedMapping,
        2, enummembers_FbxWeightedMapping,
        0, 0,
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    },
    doc_FbxWeightedMapping,
    -1,
    -1,
    0,
    0,
    init_type_FbxWeightedMapping,
    0,
    0,
#if PY_MAJOR_VERSION >= 3
    0,
    0,
#else
    0,
    0,
    0,
    0,
#endif
    dealloc_FbxWeightedMapping,
    0,
    0,
    0,
    release_FbxWeightedMapping,
    0,
    0,
    0,
    0,
    0,
    0,
    0
};
